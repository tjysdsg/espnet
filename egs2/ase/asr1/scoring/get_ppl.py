"""
Get speechocean762 perceived phones

If the score of a phone is 1 or 2, the label is used as is
If the score is 0, the label is set to SPN
"""
import argparse
from speechocean762 import load_human_scores
from utils import remove_consecutive_phone


def get_args():
    parser = argparse.ArgumentParser(description='Get speechocean762 perceived phones')
    parser.add_argument('--scores', type=str, default=None, help='Path to scores.json')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    scores, phones = load_human_scores(args.scores)
    for utt, scs in scores.items():
        phns = phones[utt]
        n = len(phns)
        for i in range(n):
            if scs[i] == 0:
                phns[i] = 'SPN'

        phns = remove_consecutive_phone(phns, 'SPN')
        p_str = ' '.join(phns)
        print(f'{utt}\t{p_str}')


if __name__ == '__main__':
    main()
