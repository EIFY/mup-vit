# Modified from https://github.com/pytorch/examples/blob/main/imagenet/main.py

import argparse
import functools
import math
import os
import pickle
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

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Subset

import wandb

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
import torch.nn.functional as F


# Taken from https://github.com/lucidrains/vit-pytorch, likely ported from https://github.com/google-research/big_vision/
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class ScaledGELU(nn.Module):
    def forward(self, input: torch.Tensor):
        return math.sqrt(2) * F.gelu(input)


class MLPBlock(nn.Sequential):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, bias: bool):
        layers = [
            nn.Linear(in_dim, mlp_dim, bias=bias),
            ScaledGELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, in_dim, bias=bias),
            nn.Dropout(dropout),
        ]
        super().__init__(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class SelfAttention(nn.Module):
    """Muon-friendly with merged QKV weights"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Follow big_vision's init
        bound = math.sqrt(3 / hidden_dim)
        self.qkv_w = nn.Parameter(torch.empty(3, hidden_dim, hidden_dim).uniform_(-bound, bound))
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        nn.init.uniform_(self.out.weight, -bound, bound)

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, -1).chunk(3, dim=-2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.dropout).transpose(1, 2)
        y = y.contiguous().view(B, T, self.hidden_dim)
        y = self.out(y)
        return y


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        bias : bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, dropout=attention_dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, bias)


    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        bias : bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                bias,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))


def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def _learned_embeddings(self, num):
        return nn.Parameter(torch.normal(mean=0., std=math.sqrt(1 / self.hidden_dim), size=(1, num, self.hidden_dim)))

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        posemb: str = "sincos2d",
        representation_size: Optional[int] = None,
        pool_type: str = "gap",
        register: int = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.RMSNorm, eps=1e-6, elementwise_affine=False),
        bias: bool = True,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.pool_type = pool_type
        self.norm_layer = norm_layer
        self.register = register + (pool_type == 'tok')  # [CLS] token is just another register
        if self.register == 1:
            self.register_buffer("reg", torch.zeros(1, 1, hidden_dim))
        elif self.register > 1:  # Random initialization needed to break the symmetry
            self.reg = self._learned_embeddings(self.register)

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        h = w = image_size // patch_size
        seq_length = h * w
        if posemb == "sincos2d":
            self.register_buffer("pos_embedding", posemb_sincos_2d(h=h, w=w, dim=hidden_dim))
        elif posemb == "learn":
            self.pos_embedding = self._learned_embeddings(seq_length)
        else:
            self.pos_embedding = None

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            bias=bias,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes, bias=bias)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size, bias=bias)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes, bias=bias)

        self.heads = nn.Sequential(heads_layers)

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1] // self.conv_proj.groups
        jax_lecun_normal(self.conv_proj, fan_in)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            jax_lecun_normal(self.heads.pre_logits, fan_in)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            if self.heads.head.bias is not None:
                nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        if self.register:
            n = x.shape[0]
            x = torch.cat([torch.tile(self.reg, (n, 1, 1)), x], dim=1)
        x = self.encoder(x)
        if self.pool_type == 'tok':
            x = x[:, 0]
        else:
            x = x[:, self.register:]
            x = x.mean(dim = 1)
        # "headless" forward pass
        if self.representation_size is not None:
            x = self.heads.pre_logits(x)
            x = self.heads.act(x)
        # Rademacher vector
        random_v = 2 * torch.randint_like(x, low=0, high=2) - 1
        return torch.sum(x * random_v)


# "(...)/python3.10/site-packages/torch/_inductor/compile_fx.py:140: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--prefetch-factor', default=1, type=int, metavar='N',
                    help='number of batches for each worker to prefetch (default: 1)')
parser.add_argument('--hidden-dim', default=384, type=int, metavar='N',
                    help='Embedding dimension of the ViT (default: 384)')
parser.add_argument('--input-resolution', default=224, type=int, metavar='RES',
                    help='Input resolution, i.e. train/val crop size (default: 224)')
parser.add_argument('--patch-size', default=16, type=int, metavar='PS')
parser.add_argument('--num-layers', default=12, type=int, metavar='N')
parser.add_argument('--num-heads', default=6, type=int, metavar='N')
parser.add_argument('--posemb', default='sincos2d', type=str,
                    choices=['none', 'sincos2d', 'learn'])
parser.add_argument('--mlp-head', action='store_true',
                    help='Use a MLP classification head with one hidden tanh layer '
                         'instead of a single linear layer')
parser.add_argument('--representation-size', default=None, type=int, metavar='N',
                    help='Size of the MLP classification head hidden layer, '
                         "defaults to --hidden-dim. No effect if --mlp-head isn't set")
parser.add_argument('--pool-type', default='gap', type=str, choices=['gap', 'tok'])
parser.add_argument('--register', default=0, type=int, metavar='N',
                    help='Number of registers (additional tokens), see '
                         'https://arxiv.org/abs/2309.16588')
parser.add_argument('--epochs', '--ep', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-steps', default=2500, type=int, metavar='N',
                    help='eval and log every N steps')
parser.add_argument('--log-epoch', nargs='*', default=[], type=int,
                    help='eval and log at the specified epochs.')
parser.add_argument('--start-step', default=0, type=int, metavar='N',
                    help='manual step number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--accum-freq", default=1, type=int,
                    help="Update the model every --acum-freq steps.")
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='maximum learning rate', dest='lr')
parser.add_argument('--min-ratio', default=0., type=float,
                    help='minimum LR ratio at the end of decay')
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--momentum', '--mo', default=0.1, type=float,
                    help='momentum for non-sign parameters')
parser.add_argument('--end-mo-ratio', default=1.0, type=float)
parser.add_argument('--cautious', action='store_true',
                    help='Cautious weight decay (https://arxiv.org/abs/2510.12402v1)')
parser.add_argument('--cos-power', default=1.0, type=float,
                    help='power of the cosine LR decay, defaults to 1')
parser.add_argument('--power', default=None, type=float,
                    help='power of the polynomial LR decay, defaults to cosine LR decay')
parser.add_argument('--sign-lr', default=0.2, type=float,
                    help='maximum learning rate for the output layer')
parser.add_argument('--corrected', action='store_true')
parser.add_argument('--wd', '--weight-decay', default=0.08, type=float)
parser.add_argument('--c-sq', default=1.1875, type=float,
                    help='normalized steady-state norm squared for spectral parameters.')
parser.add_argument('--bias-wd', default=math.inf, type=float)
parser.add_argument('--bias-c-sq', default=0., type=float)
parser.add_argument('--sign-weight-decay', '--sign-wd', default=0.004, type=float,
                    help='sign weight decay (default: 0.004)')
parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                    help="Max norm for gradient clip (default: 1.0)")
parser.add_argument('--torchvision-inception-crop', action='store_true',
                    help="Switch back to torchvision's RandomResizedCrop(), "
                         'which actually improves the model')
parser.add_argument('--lower-scale', type=float, default=0.05,
                    help="Lower bound of the area of the Inception crop (default: 0.05)")
parser.add_argument('--upper-scale', type=float, default=1.0,
                    help="Upper bound of the area of the Inception crop (default: 1.0)")
parser.add_argument('--mixup-alpha', default=0.2, type=float,
                    help='Beta distribution shape parameter for the MixUp (default: 0.2). '
                         'Use 0.0 to turn MixUp off.')
parser.add_argument('--randaug', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='Use RandAug (default: True)')
parser.add_argument("--randaug-magnitude", default=10, type=int)
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
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fake-data', action='store_true', help="use fake data to benchmark")
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


def collate(batch, mixup):
    return mixup(*torch.utils.data.default_collate(batch))


def chunk(n, device, *tensors):
    tensors = tuple(t.to(device=device, non_blocking=True) for t in tensors)
    if n == 1:
        yield tensors
    else:
        yield from zip(*(t.chunk(n) for t in tensors))


def norm_info(model):
    count = 0
    hidden = 0.
    for n, p in model.named_parameters():
        if not n.endswith("heads.head.weight"):
            count += p.numel()
            hidden += torch.sum(p ** 2).item()
    return count, hidden


def main():
    args = parser.parse_args()

    if not args.mlp_head:
        args.representation_size = None
    elif args.representation_size is None:
        args.representation_size = args.hidden_dim

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
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

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
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args)


def is_primary(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')


# None is tombstone value, '' (empty string) is for store_true flags
# Best hyperparameters w/ cosine LR schedule, taken from corrected_c_sq_lr.sh
default = {'corrected': '', 'ep': 90, 'momentum': 0.1, 'lr': 0.011584472366059664, 'sign_lr': 0.1, 'c_sq': 0.8396893026590251, 'wd': None, 'sign_wd': 0.00282842712474619, 'nesterov': '', 'cos_power': None, 'power': None}


def run_name(opt, d, repeat=0):
    l = [opt]
    for k, v in d.items():
        if v is not None:
            l.append(k)
            if v != '':
                if type(v) is float:
                    v = f"{v:.3g}"
                else:
                    v = str(v)
                l.append(v)
    l.append(str(repeat))
    return '-'.join(l)


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
    if args.distributed or args.ngpus_per_node > 1:
        torch.cuda.set_device(args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    zero_bias_norm = (
        args.corrected and args.bias_c_sq == 0.) or (
        not args.corrected and args.bias_wd == math.inf)
    args.bias = not zero_bias_norm

    # Create model
    # We keep the original model and use it to save checkpoints or access submodules since:
    # 1. PyTorch 2.0+ adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # 2. DDP wraps the model as the "module" attribute.

    original_model = model = SimpleVisionTransformer(
        image_size=args.input_resolution,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.hidden_dim * 4,
        posemb=args.posemb,
        representation_size=args.representation_size,
        pool_type=args.pool_type,
        register=args.register,
        bias=args.bias,
    )

    args.total_batch_size = args.batch_size

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        model.cuda()
        device = torch.device("cuda")
        if args.distributed or args.ngpus_per_node > 1:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available
            if args.gpu is not None:
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / args.ngpus_per_node)
                args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        device = torch.device("mps")
        model = model.to(device)

    # Data loading code
    if args.fake_data:
        print("=> Fake data is used!")
        input_shape = (3, args.input_resolution, args.input_resolution)
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        val_dataset = datasets.FakeData(50000, input_shape, 1000, transform)
    else:
        value_range = v2.Normalize(
            mean=[0.5] * 3,
            std=[0.5] * 3)
        val_dataset = datasets.ImageNet(
            args.data,
            split='val',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(args.input_resolution),
                v2.ToDtype(torch.float32, scale=True),
                value_range,
            ]))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.prefetch_factor * args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        multiprocessing_context='spawn', prefetch_factor=1)

    print('Compiling model...')

    # Inductor doesn't support MPS yet (https://github.com/pytorch/pytorch/issues/125254)
    model = torch.compile(model, backend="aot_eager" if device.type == 'mps' else "inductor")

    curr = dict(default)
    factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
    opt = 'scion-t212'
    res = {}

    prefix = "module."
    pre_len = len(prefix)
    for lr_f in factors:
        for c_sq_f in factors:
            curr['lr'] = lr_f * math.sqrt(c_sq_f) * default['lr']
            curr['c_sq'] = c_sq_f * default['c_sq']
            name = run_name(opt, curr)
            args.resume = f"logs/{name}/checkpoints/model_step_28151.pth.tar"
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            args.start_step = checkpoint['step']
            best_acc1 = checkpoint['best_acc1']
            state_dict = checkpoint['state_dict']
            if any(k.startswith(prefix) for k in state_dict):  # Old code DPP model state_dict
                state_dict = {k[pre_len:]: v for k, v in state_dict.items()}
            original_model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.resume, checkpoint['step']))

            count, hidden = norm_info(original_model)
            hidden **= 0.5
            print(f"{count=}, {hidden=}")
            jacobian = estimate_jacobian(val_loader, model, device, args)
            print(f"{jacobian=}")
            res[name] = dict(hidden=hidden, jacobian=jacobian)

    filename = 'vit_jacobian_norms.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(res, file)

    if args.distributed or args.ngpus_per_node > 1:
        dist.destroy_process_group()


def estimate_jacobian(val_loader, model, device, args):
    # switch to evaluate mode
    model.eval()
    torch.cuda.empty_cache()
    squared_total = 0.0
    n = 0
    # generator moves data to the same device as model
    gen = (b for images, target in val_loader for b in chunk(args.prefetch_factor, device, images, target))
    for images, target in gen:
        for img, _ in chunk(args.accum_freq, device, images, target):
            # compute output
            loss = model(img)
            loss.backward()
            # Hutchinson's trace estimator for the Frobenius norm of the (img.shape[0] * args.hidden_dim, n_of_paramters) Jacobian matrix
            l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm).item()
            squared_total += l2_grads ** 2
            n += img.size(0)
            model.zero_grad()
    print(f"{n=}")
    # Mean (args.hidden_dim, n_of_paramters) Jacobian matrix Frobenius norm
    jacobian = math.sqrt(squared_total / n)
    return jacobian


if __name__ == '__main__':
    main()
