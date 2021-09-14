import argparse
import os
from speechocean762 import load_utt2cpl_ppl_scores
from utils import write_utt2seq


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=str, help='Path to scores.json')
    parser.add_argument('--output-dir', type=str, help='Output dir')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    utt2cpl, utt2ppl, utt2scores = load_utt2cpl_ppl_scores(args.scores, floor=1)

    # convert ppls to score phones
    for utt in utt2ppl.keys():
        scores = utt2scores[utt]
        phones = utt2ppl[utt]
        for i, p in enumerate(phones):  # type: int, str
            s = int(scores[i])
            if '*' in p:  # score 1 if accented
                s = 1
                p = p.replace('*', '')

            # this must be after if '*' in p, cuz 0-score with phone* might occur in scores.json
            if s == 0:  # e.g. AH0 -> EH2 where EH is the true pronunciation
                s = 2

            score_phone = f'{p}{s}'

            if '<UNK>' in score_phone.upper():  # <UNK> -> SPN without a score
                score_phone = 'SPN'

            utt2ppl[utt][i] = score_phone

        # remove <DEL>
        utt2ppl[utt] = [p for p in utt2ppl[utt] if "<DEL>" not in p]

    write_utt2seq(os.path.join(output_dir, 'cpl.txt'), utt2cpl)
    write_utt2seq(os.path.join(output_dir, 'text'), utt2ppl)
    write_utt2seq(os.path.join(output_dir, 'utt2scores'), utt2scores)


if __name__ == '__main__':
    main()
