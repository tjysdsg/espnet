"""
@file Generate kaldi format of aishell-ssb
"""

from generate_phoneme_transcript import generate_phoneme_transcript
import argparse
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    generate_phoneme_transcript(args.out_dir)  # text

    utt2spk = open(os.path.join(out_dir, 'utt2spk'), 'w')
    wavscp = open(os.path.join(out_dir, 'wav.scp'), 'w')
    with open(os.path.join(FILE_DIR, 'aishell-ssb-annotations.txt'), encoding='utf-8') as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0].split('.')[0]
            spk = utt[:6]

            # utt2spk
            utt2spk.write(f'{utt}\t{spk}\n')
            # wav.scp
            path = os.path.join(args.wav_dir, spk, f'{utt}.wav')
            wavscp.write(f'{utt}\t{path}\n')

    utt2spk.close()
    wavscp.close()


if __name__ == '__main__':
    main()
