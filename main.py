# Modified from https://github.com/pytorch/examples/blob/main/imagenet/main.py

import argparse
import os
import math
import random
import shutil
import time
import warnings
from datetime import datetime
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Subset

from simple_vit import SimpleVisionTransformer

import wandb

# "(...)/python3.10/site-packages/torch/_inductor/compile_fx.py:140: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-steps', default=2500, type=int, metavar='N',
                    help='eval and log every N steps')
parser.add_argument('--start-step', default=0, type=int, metavar='N',
                    help='manual step number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--accum-freq", default=1, type=int,
                    help="Update the model every --acum-freq steps.")
parser.add_argument("--warmup", default=10000, type=int,
                    help="Number of steps to warmup for.")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='maximum learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 0.1)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument("--logs", default="./logs/", type=str,
                    help="Where to store logs. Use None to avoid storing logs.")
parser.add_argument('--name', default=None, type=str,
                    help='Optional identifier for the experiment when storing logs. '
                         'Otherwise use current time.')
parser.add_argument("--report-to", default='', type=str,
                    help="Options are ['wandb']")
parser.add_argument("--wandb-notes", default='', type=str,
                    help="Notes if logging with wandb")
best_acc1 = 0


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.horovod:
        return hvd.broadcast_object(obj, root_rank=src)
    else:
        if args.rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        args.ngpus_per_node = torch.cuda.device_count()
        if args.ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        args.ngpus_per_node = 1

    # get the name of the experiments
    if args.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
        ])

    log_base_path = os.path.join(args.logs, args.name)
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.wandb = 'wandb' in args.report_to

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, ))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args)


def is_master(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % arg.ngpus_per_node == 0)


def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = SimpleVisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
    )

    wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
    non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / args.ngpus_per_node)
                args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": wd_params, "weight_decay": args.weight_decay},
            {"params": non_wd_params, "weight_decay": 0.},
        ],
        lr=args.lr,
    )

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000,
            v2.ToDtype(torch.float32, scale=True))
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000,
            v2.ToDtype(torch.float32, scale=True))
    else:
        value_range = v2.Normalize(
            mean=[0.5] * 3,
            std=[0.5] * 3)

        train_dataset = datasets.ImageNet(
            args.data,
            split='train',
            transform=v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(224, scale=(0.05, 1.0), antialias=False),
                v2.RandomHorizontalFlip(),
                v2.RandAugment(2, 10, fill=[128] * 3),
                v2.ToDtype(torch.float32, scale=True),
                value_range,
            ]))

        val_dataset = datasets.ImageNet(
            args.data,
            split='val',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Resize(256, antialias=False),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                value_range,
            ]))

    n = len(train_dataset)
    total_steps = round(n * args.epochs / args.batch_size)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    mixup = v2.MixUp(alpha=0.2, num_classes=1000)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        collate_fn=lambda batch: mixup(*torch.utils.data.default_collate(batch)), drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step / args.warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [args.warmup])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_step = checkpoint['step']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.resume, checkpoint['step']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.wandb and is_master(args):
        wandb.init(
            project="mup-vit",
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto',
            config=vars(args),
        )
        params_file = os.path.join(args.logs, args.name, "params.txt")
        wandb.save(params_file)

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    print('Compiling model...')
    original_model = model
    model = torch.compile(original_model)

    if args.evaluate:
        # evaluate on validation set.
        # I got RuntimeError: Found a custom (non-ATen) operator that either mutates or its inputs: aten::record_stream..
        # if I use the compiled model, so for now I pass in original_model instead.
        validate(val_loader, original_model, criterion, args.start_step, args)
        return

    train(train_loader, val_loader, args.start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device, args)


def train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, losses, top1, top5])

    # switch to train mode
    model.train()
    end = time.time()
    best_acc1 = 0

    def infinite_loader():
        while True:
            yield from train_loader

    for step, (images, target) in zip(range(start_step + 1, total_steps + 1), infinite_loader()):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        step_loss = step_acc1 = step_acc5 = 0.0

        for img, trt in zip(images.chunk(args.accum_freq), target.chunk(args.accum_freq)):
            # compute output
            output = model(img)
            loss = criterion(output, trt)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, trt, topk=(1, 5), class_prob=True)
            step_loss += loss.item()
            step_acc1 += acc1[0].item()
            step_acc5 += acc5[0].item()

            # compute gradient
            (loss / args.accum_freq).backward()

        step_loss /= args.accum_freq
        step_acc1 /= args.accum_freq
        step_acc5 /= args.accum_freq

        losses.update(step_loss, images.size(0))
        top1.update(step_acc1, images.size(0))
        top5.update(step_acc5, images.size(0))

        # do SGD step
        l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            progress.display(step)
            if args.wandb and is_master(args):

                with torch.no_grad():
                    l2_params = sum(p.square().sum().item() for _, p in model.named_parameters())

                samples_per_second_per_gpu = args.batch_size / batch_time.val
                samples_per_second = samples_per_second_per_gpu * args.world_size
                log_data = {
                    "train/loss": step_loss,
                    'train/acc@1': step_acc1,
                    'train/acc@5': step_acc5,
                    "data_time": data_time.val,
                    "batch_time": batch_time.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": scheduler.get_last_lr()[0],
                    "l2_grads": l2_grads.item(),
                    "l2_params": math.sqrt(l2_params)
                }
                wandb.log(log_data, step=step)

        if step % args.log_steps == 0 or step == total_steps:

            acc1 = validate(val_loader, original_model, criterion, step, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_master(args):
                save_checkpoint({
                    'step': step,
                    'state_dict': original_model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best, args.checkpoint_path)

        scheduler.step()


def validate(val_loader, model, criterion, step, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            torch.cuda.empty_cache()
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                for img, trt in zip(images.chunk(args.accum_freq), target.chunk(args.accum_freq)):
                    # compute output
                    output = model(img)
                    loss = criterion(output, trt)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, trt, topk=(1, 5))
                    losses.update(loss.item(), img.size(0))
                    top1.update(acc1[0].item(), img.size(0))
                    top5.update(acc5[0].item(), img.size(0))
                    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    if args.wandb and is_master(args):        
        log_data = {
            'val/loss': losses.avg,
            'val/acc@1': top1.avg,
            'val/acc@5': top5.avg,
        }
        wandb.log(log_data, step=step)

    return top1.avg


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,), class_prob=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # with e.g. MixUp target is now given by probabilities for each class so we need to convert to class indices
        if class_prob:
            _, target = target.topk(1, 1, True, True)
            target = target.squeeze(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
