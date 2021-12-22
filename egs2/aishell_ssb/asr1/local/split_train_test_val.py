"""
@file Split generated text/utt2spk/wav.scp into train/test/val dirs
"""

import argparse
import os
from utils import get_spk_from_utt

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    train_dir = os.path.join(out_dir, 'train')
    test_dir = os.path.join(out_dir, 'test')
    val_dir = os.path.join(out_dir, 'val')
    for d in [train_dir, test_dir, val_dir]:
        os.makedirs(d, exist_ok=True)

    # read {train,test,val}_spk.txt
    with open(os.path.join(FILE_DIR, 'train_spk.txt'), encoding='utf-8') as f:
        train_spks = set([line.strip('\n') for line in f])
    with open(os.path.join(FILE_DIR, 'test_spk.txt'), encoding='utf-8') as f:
        test_spks = set([line.strip('\n') for line in f])
    with open(os.path.join(FILE_DIR, 'val_spk.txt'), encoding='utf-8') as f:
        val_spks = set([line.strip('\n') for line in f])

    # utt2spk
    for name in ['utt2spk', 'text', 'wav.scp']:
        train_of = open(os.path.join(train_dir, name), 'w', encoding='utf-8')
        test_of = open(os.path.join(test_dir, name), 'w', encoding='utf-8')
        val_of = open(os.path.join(val_dir, name), 'w', encoding='utf-8')
        with open(os.path.join(in_dir, name), encoding='utf-8') as f:
            for line in f:
                utt, sth = line.strip('\n').split(maxsplit=1)

                spk = get_spk_from_utt(utt)
                if spk in train_spks:
                    train_of.write(f'{utt}\t{sth}\n')
                elif spk in test_spks:
                    test_of.write(f'{utt}\t{sth}\n')
                elif spk in val_spks:
                    val_of.write(f'{utt}\t{sth}\n')

        train_of.close()
        test_of.close()
        val_of.close()


if __name__ == '__main__':
    main()
