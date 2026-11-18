"""Convert a checkpoint saved by a PAI run of main.py into a plain SimpleVisionTransformer
checkpoint that --pai-init can load.

Only valid for a checkpoint saved before PAI's first switch: neuron weights are copied, PAI
bookkeeping (tracker_string, dendrite_module.*, shape buffers) is dropped, and any dendrite
weights would be dropped silently too. Use it to branch several
perforation arms from one neuron-phase checkpoint.

    python scripts/export_plain_checkpoint.py RUN/archive/SNAPSHOT/checkpoint.pth.tar OUT.pth.tar \
        --hidden-dim 192 --num-layers 12 --num-heads 3 --mlp-head --pai-target mlp
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pai_setup import plain_state_dict  # noqa: E402
from simple_vit import SimpleVisionTransformer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=3)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--input-resolution', type=int, default=224)
    parser.add_argument('--mlp-head', action='store_true')
    parser.add_argument('--pai-target', default=None, choices=['mlp', 'ln_mlp', 'ln_fc1', 'fc1', 'fc2'],
                        help='wrapping the checkpoint was saved with (default: detected)')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu', weights_only=True)
    source = checkpoint['state_dict']
    state = plain_state_dict(source, args.pai_target)

    model = SimpleVisionTransformer(
        image_size=args.input_resolution, patch_size=args.patch_size, num_layers=args.num_layers,
        num_heads=args.num_heads, hidden_dim=args.hidden_dim, mlp_dim=args.hidden_dim * 4,
        representation_size=args.hidden_dim if args.mlp_head else None)
    # strict=True: every plain parameter must come from the checkpoint with the right shape.
    model.load_state_dict(state, strict=True)

    torch.save({'step': checkpoint['step'], 'best_acc1': checkpoint['best_acc1'],
                'state_dict': model.state_dict(), 'source': os.path.abspath(args.src)}, args.dst)
    print(f"=> wrote {args.dst}: {len(state)} tensors from step {checkpoint['step']} "
          f"(best acc@1 {checkpoint['best_acc1']:.4f})")


if __name__ == '__main__':
    main()
