"""
Generate utt2scores from text
"""
import argparse
import copy
import random
import os
from utils import load_utt2phones, load_utt2seq, write_utt2seq
from typing import List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='text file')
    parser.add_argument('--output-path', type=str, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    utt2phones = load_utt2phones(args.text)

    of = open(args.output_path, 'w')
    for utt, phones in utt2phones.items():
        scores = ['2' for _ in phones]
        of.write(f'{utt}\t{" ".join(scores)}\n')

    of.close()


if __name__ == '__main__':
    main()
