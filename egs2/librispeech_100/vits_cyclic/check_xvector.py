"""
Check whether x-vectors in --dir1 and --dir2 are the same. The x-vectors are listed in *.scp files under --dir1 and --dir2.
"""
import kaldiio
import numpy as np
from argparse import ArgumentParser
import os
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dir1', type=str, required=True)
    parser.add_argument('--dir2', type=str, required=True)
    return parser.parse_args()


def read_all_xvectors(folder: str):
    ret = {}
    for file in os.listdir(folder):
        if file.endswith(".scp"):
            with open(os.path.join(folder, file)) as f:
                for line in f:
                    key, ark_path = line.rstrip('\n').split()
                    ark_path = os.path.join(folder, Path(ark_path).name)
                    # print(key, ark_path)

                    arr = kaldiio.load_mat(ark_path)
                    ret[key] = arr
    return ret


def main():
    args = get_args()

    utt2vec1 = read_all_xvectors(args.dir1)
    utt2vec2 = read_all_xvectors(args.dir2)

    keys = set(list(utt2vec1.keys()) + list(utt2vec2.keys()))

    for k in keys:
        if k not in utt2vec1:
            continue
        v1 = utt2vec1[k]
        if k not in utt2vec2:
            continue
        v2 = utt2vec2[k]

        # print(k)
        if not np.allclose(v1, v2):
            print(f'utt {k} has different x-vector')


if __name__ == '__main__':
    main()
