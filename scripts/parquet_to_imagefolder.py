"""Unpack the evanarlian/imagenet_1k_resized_256 parquet shards into the ImageFolder layout
main.py reads: OUT/{train,val}/<wnid>/<original filename>. JPEG bytes are written unchanged.

    python scripts/parquet_to_imagefolder.py SRC_DIR OUT_DIR --workers 16

SRC_DIR holds the train-*.parquet and val-*.parquet shards. Class folders are WordNet IDs taken
from the original filenames, so ImageFolder's sorted class order matches the dataset's labels
(checked below) and runs see the same class indices as on PSC and AWS.
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq

WNID = re.compile(r"n\d{8}")
OUT = None


def convert(path):
    split = "train" if os.path.basename(path).startswith("train-") else "val"
    pf = pq.ParquetFile(path)
    n, mapping = 0, {}
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i)
        for img, label in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            name = os.path.basename(img["path"])
            wnid = WNID.findall(name)[-1]
            if mapping.setdefault(label, wnid) != wnid:
                raise ValueError(f"label {label} maps to {mapping[label]} and {wnid}")
            d = os.path.join(OUT, split, wnid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "wb") as f:
                f.write(img["bytes"])
            n += 1
    return split, n, mapping


def init(out):
    global OUT
    OUT = out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    if not files:
        sys.exit(f"no parquet files in {args.src}")
    counts, mapping = {"train": 0, "val": 0}, {}
    with Pool(args.workers, initializer=init, initargs=(args.out,)) as pool:
        for split, n, m in pool.imap_unordered(convert, files):
            counts[split] += n
            for k, v in m.items():
                if mapping.setdefault(k, v) != v:
                    sys.exit(f"inconsistent label {k}: {mapping[k]} vs {v}")
    wnids = [mapping[k] for k in sorted(mapping)]
    if len(wnids) != 1000 or wnids != sorted(wnids):
        sys.exit("labels do not follow sorted-WNID order; ImageFolder class indices would differ")
    print(f"=> unpacked {counts} into {args.out} (label order matches ImageFolder)")


if __name__ == "__main__":
    main()
