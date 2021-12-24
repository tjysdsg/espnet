"""
@file Generate kaldi format of aishell-ssb
"""

from generate_phoneme_transcript import generate_phoneme_transcript
import argparse
import os
from utils import get_spk_from_utt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/storage20/dinkelheinrich/data/AISHELL3')
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    generate_phoneme_transcript(args.data_dir, args.out_dir)  # text

    for data_split in ['train', 'test']:
        in_dir = os.path.join(args.data_dir, data_split)

        split_dir = os.path.join(args.out_dir, data_split)
        os.makedirs(split_dir, exist_ok=True)

        utt2spk = open(os.path.join(split_dir, 'utt2spk'), 'w')
        wavscp = open(os.path.join(split_dir, 'wav.scp'), 'w')
        with open(os.path.join(in_dir, 'content.txt'), encoding='utf-8') as f:
            for line in f:
                tokens = line.strip('\n').split()
                utt = tokens[0].split('.')[0]
                spk = get_spk_from_utt(utt)

                # utt2spk
                utt2spk.write(f'{utt}\t{spk}\n')

                # wav.scp
                path = os.path.join(in_dir, 'wav', spk, f'{utt}.wav')
                wavscp.write(f'{utt}\t{path}\n')

        utt2spk.close()
        wavscp.close()


if __name__ == '__main__':
    main()
