import argparse
import os
from speechocean762 import load_human_scores, load_so762_ref
from utils import write_utt2seq


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=str, help='Path to scores.json')
    parser.add_argument('--text-phone', type=str, help='Path to the text-phone file')
    parser.add_argument('--output-dir', type=str, help='Path to the text-phone file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    utt2phones = load_so762_ref(args.text_phone)
    utt2scores, _ = load_human_scores(args.scores, floor=1)

    # convert phones to score phones
    for utt in utt2phones.keys():
        scores = utt2scores[utt]
        phones = utt2phones[utt]
        for i, p in enumerate(phones):
            utt2phones[utt][i] = f'{p}{scores[i]}'

    write_utt2seq(os.path.join(output_dir, 'text'), utt2phones)
    write_utt2seq(os.path.join(output_dir, 'utt2scores'), utt2scores)


if __name__ == '__main__':
    main()
