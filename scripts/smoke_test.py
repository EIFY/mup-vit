"""Check that this machine can run PAI training, without starting a real run.

    python scripts/smoke_test.py [--data DIR] [--layers 11] [--target ln_mlp] [--offline]

1. GPU: torch sees it, this torch build has kernels for it, and an FP32 forward/backward runs.
2. Credentials: WANDB_API_TOKEN, PAIEMAIL and PAITOKEN are set; the W&B key is accepted.
3. Perforation: ViT-Tiny is perforated exactly as main.py does it, and the report lists which
   encoder blocks got dendrites, so you can confirm the layer choice.
4. Candidate scores: one Perforated Backpropagation dendrite epoch on random data, after which
   the candidates must have nonzero correlation scores that PAI counts as improvements. This is
   what failed with perforatedai 4.0.0: its pai_improvement_threshold_raw of 1e-4 is far above
   the raw covariance scores (~1e-12 here), so no candidate ever improved and the PB scores
   stayed 0. pai_setup.py sets the threshold to 0; this check also fails if it is not 0.
5. Dataset (with --data): DIR/train and DIR/val exist with matching class folders, e.g. after a
   job has unpacked the parquet shards to local disk.

Expects the credentials in the environment (setup_machine.sh exports .env before calling this).
Exits non-zero on the first failed check.
"""
import argparse
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def section(title):
    print(f'\n=== {title} ===', flush=True)


def fail(message):
    sys.exit(f'SMOKE TEST FAILED: {message}')


def check_gpu():
    import torch
    section('GPU')
    if not torch.cuda.is_available():
        fail('torch.cuda.is_available() is False (no GPU, or driver/torch mismatch)')
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f'sm_{major}{minor}'
    print(f'{name}, {arch}, torch {torch.__version__} (CUDA {torch.version.cuda})')
    # Kernels built for sm_XY also run on sm_XZ with Z >= Y (same major), so e.g. the RTX 4090 and
    # L40S (sm_89) use the sm_86 kernels; PyTorch's wheels ship no separate sm_89 build.
    # 'sm_86' -> (8, 6), 'sm_120' -> (12, 0)
    built = [(int(a[3:-1]), int(a[-1]), a) for a in torch.cuda.get_arch_list()
             if a.startswith('sm_') and a[3:].isdigit()]
    compatible = [a for M, m, a in sorted(built) if M == major and m <= minor]
    if compatible:
        print(f'kernels: {compatible[-1]}' + ('' if compatible[-1] == arch else f' (compatible with {arch})'))
    else:
        fail(f'this torch build has no kernels for {arch} (it has {torch.cuda.get_arch_list()}); '
             're-run setup_machine.sh so it picks a matching torch build')
    # Training runs in FP32 with TF32 tensor cores (main.py --tf32), as the run script does.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    x = torch.randn(256, 256, device='cuda', requires_grad=True)
    y = (x @ x).relu().sum()
    y.backward()
    torch.cuda.synchronize()
    print('FP32/TF32 forward/backward ok')


def check_credentials(offline):
    section('Credentials')
    missing = [k for k in ('WANDB_API_TOKEN', 'PAIEMAIL', 'PAITOKEN') if not os.environ.get(k)]
    if missing:
        fail(f'not set: {", ".join(missing)} (fill them in .env)')
    print('WANDB_API_TOKEN, PAIEMAIL, PAITOKEN are set')
    if offline:
        print('--offline: skipping the W&B key check')
        return
    import wandb
    # Api().viewer checks the key without writing it to ~/.netrc, unlike wandb.login().
    try:
        user = wandb.Api(api_key=os.environ['WANDB_API_TOKEN']).viewer.username
    except Exception as e:
        fail(f'W&B rejected WANDB_API_TOKEN ({type(e).__name__})')
    print(f'W&B key accepted: user {user}, logging to entity {os.environ.get("WANDB_ENTITY") or "(default)"}')


def check_perforation(layers, target):
    import torch
    import main
    import pai_setup
    from simple_vit import SimpleVisionTransformer
    section(f'Perforation: --pai-layers {" ".join(map(str, layers))} --pai-target {target}')
    args = main.parser.parse_args([
        '--tf32', '--mlp-head', '--hidden-dim', '192', '--num-layers', '12', '--num-heads', '3',
        '--pai-layers', *map(str, layers), '--pai-target', target,
        '--pai-initial-correlation-batches', '2'])
    args.pai = True
    model = SimpleVisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=3,
                                    hidden_dim=192, mlp_dim=768, representation_size=192)
    # PAI writes ./pai (config dump); keep it out of the repo. Its license check happens here.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            model = pai_setup.perforate(model, args)
        finally:
            os.chdir(cwd)

    perforated = []
    for i, block in enumerate(model.encoder.layers):
        dendrite_modules = [n for n, m in block.named_modules() if type(m).__name__ == 'PAIDendriteModule']
        if dendrite_modules:
            perforated.append(i)
        wrapped = ', '.join(n.split('.')[0] for n in dendrite_modules) or '-'
        print(f'  encoder layer {i:2d}: {"DENDRITES on " + wrapped if dendrite_modules else "tracked only"}')
    if perforated != sorted(layers):
        fail(f'expected dendrites on layers {sorted(layers)}, got {perforated}')
    print(f'\nWrapped unit of layer {layers[0]}:')
    print(model.encoder.layers[layers[0]].mlp if target in ('mlp', 'ln_mlp') else model.encoder.layers[layers[0]])

    model = model.cuda().train()
    x = torch.randn(8, 3, 224, 224, device='cuda')
    y = torch.randint(0, 1000, (8,), device='cuda')
    _, loss = model(x, 1.0, y, y)
    loss.backward()
    model.zero_grad()
    print(f'perforated forward/backward ok (loss {loss.item():.3f})')
    return model, args


def check_candidate_scores(model, args):
    """One PB dendrite epoch; the candidates' correlation scores must register as improvements."""
    import torch
    import pai_setup
    from perforatedai import globals_perforatedai as GPA
    section('Candidate dendrite scores (Perforated Backpropagation)')
    threshold = GPA.pc.get_pai_improvement_threshold_raw()
    print(f'pai_improvement_threshold_raw = {threshold}')
    if threshold != 0:
        fail(f'pai_improvement_threshold_raw is {threshold}, not 0: candidate scores are tiny '
             '(~1e-12), so with this threshold none ever counts as improving')

    g = torch.Generator(device='cuda').manual_seed(0)

    def train(model, optimizer, steps):
        # As main.py does in both phases with --tf32: plain backward, then optimizer.step().
        model.train()
        for _ in range(steps):
            x = torch.randn(32, 3, 224, 224, device='cuda', generator=g)
            y = torch.randint(0, 1000, (32,), device='cuda', generator=g)
            optimizer.zero_grad()
            _, loss = model(x, 1.0, y, y)
            loss.backward()
            optimizer.step()

    # PAI writes ./pai (checkpoints, graphs) at each switch; keep it out of the repo.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            optimizer, _ = pai_setup.setup_optimizer(model, args)
            train(model, optimizer, 5)
            model, _, _ = pai_setup.add_score(model, 0.1, args, 'cuda', force_switch=True)
            if not pai_setup.in_dendrite_phase():
                fail('PAI did not switch to a dendrite phase')
            optimizer, _ = pai_setup.setup_optimizer(model, args)
            train(model, optimizer, 10)
            # Ends the dendrite epoch: PAI scores the candidates and records improvements.
            model, _, _ = pai_setup.add_score(model, 0.1, args, 'cuda')
        finally:
            os.chdir(cwd)

    for name, module in model.named_modules():
        if type(module).__name__ != 'PAIDendriteModule':
            continue
        values = module.dendrite_values[0]
        raw = values.best_score.detach().float()
        pearson = values.best_pearson_correlation.detach().float()
        improved = int(values.nodes_best_improved_this_epoch.sum().item())
        print(f'{name.removesuffix(".dendrite_module")}: {improved}/{raw.numel()} nodes improved, '
              f'max |raw score| {raw.abs().max().item():.3e}, '
              f'max |Pearson correlation| {pearson.abs().max().item():.3e}')
        if not (torch.isfinite(raw).all() and torch.isfinite(pearson).all()):
            fail('non-finite candidate scores')
        if improved == 0 or not pearson.any():
            fail('no candidate correlation score registered as an improvement '
                 '(the PB scores would stay 0, as with perforatedai 4.0.0)')
    print('candidate scores are nonzero and count as improvements')


def check_data(path):
    section(f'Dataset: {path}')
    classes = {}
    for split in ('train', 'val'):
        d = os.path.join(path, split)
        if not os.path.isdir(d):
            fail(f'{d} is missing')
        classes[split] = sorted(e.name for e in os.scandir(d) if e.is_dir())
        n_images = sum(len(os.listdir(os.path.join(d, c))) for c in classes[split])
        print(f'{split}: {len(classes[split])} classes, {n_images} images')
    if classes['train'] != classes['val']:
        fail('train and val have different class folders, so ImageFolder labels would not match')
    if len(classes['val']) != 1000:
        fail(f'expected 1000 classes, found {len(classes["val"])}')


def main_():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='ImageFolder root with train/ and val/ (optional)')
    parser.add_argument('--layers', type=int, nargs='+', default=[11])
    parser.add_argument('--target', default='ln_mlp')
    parser.add_argument('--offline', action='store_true', help='skip the W&B key check')
    args = parser.parse_args()

    check_gpu()
    check_credentials(args.offline)
    model, pai_args = check_perforation(args.layers, args.target)
    check_candidate_scores(model, pai_args)
    if args.data:
        check_data(args.data)
    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main_()
