"""
Generate data for training the scoring model using a transcript file in a standard pronunciation dataset
"""
import argparse
import copy
import random
import os
from utils import load_utt2phones
from typing import List

VOWELS = [
    'AH',
    'IH',
    'IY',
    'AE',
    'EH',
    'ER',
    'AY',
    'EY',
    'UH',
    'OY',
    'OW',
    'UW',
    'AO',
    'AA',
    'AW',
]

CONSONANTS = [
    'N',
    'T',
    'D',
    'S',
    'L',
    'R',
    'DH',
    'M',
    'Z',
    'K',
    'W',
    'HH',
    'V',
    'F',
    'P',
    'B',
    'NG',
    'G',
    'SH',
    'Y',
    'TH',
    'CH',
    'JH',
    'ZH',
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Kaldi text data file')
    parser.add_argument('--output-dir', type=str, help='Output dir')
    args = parser.parse_args()
    return args


def random_phone(orig_ph: str) -> str:
    phones = CONSONANTS
    if orig_ph in VOWELS:
        phones = VOWELS

    phones = [p for p in phones if p != orig_ph]
    return random.choice(phones)


def random_similar_phone(orig_ph: str) -> str:
    from aug import ph2similar
    phones = ph2similar.get(orig_ph, [])
    phones = [p for p in phones if p != orig_ph]
    if len(phones) > 0:
        return random.choice(phones)
    else:
        return orig_ph


def generate_sample_phones(phones: List[str], randomizer, new_score: int) -> (List[str], List[int]):
    phones = copy.deepcopy(phones)
    n = len(phones)
    scores = [2 for _ in range(n)]

    idx = random.randint(0, n - 1)
    assert phones[idx] in VOWELS + CONSONANTS  # either vowels or consonants, shouldn't be things like SIL
    new_phone = randomizer(phones[idx])
    if phones[idx] != new_phone:
        scores[idx] = new_score

    phones[idx] = new_phone
    return phones, scores


def main():
    args = get_args()
    utt2phones = load_utt2phones(args.text)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ref_file = open(os.path.join(output_dir, 'text'), 'w')
    score_file = open(os.path.join(output_dir, 'utt2scores'), 'w')
    for utt_id, phones in utt2phones.items():
        # due to data aug, an utterance id could corresponds to multiple data samples
        # so utt_id is appended by '#n'
        utt = f'{utt_id}#0'

        # generate 0-score samples
        new_phones, scores = generate_sample_phones(phones, random_phone, new_score=0)
        scores = [str(s) for s in scores]
        ref_file.write(f'{utt}\t{" ".join(new_phones)}\n')
        score_file.write(f'{utt}\t{" ".join(scores)}\n')

        utt = f'{utt_id}#1'

        # generate 1-score samples
        new_phones, scores = generate_sample_phones(phones, random_similar_phone, new_score=1)
        scores = [str(s) for s in scores]
        ref_file.write(f'{utt}\t{" ".join(new_phones)}\n')
        score_file.write(f'{utt}\t{" ".join(scores)}\n')

    ref_file.close()


if __name__ == '__main__':
    main()
