import argparse
from utils import EMPTY_PHONES, load_utt2seq, write_utt2seq
from typing import List, Dict
import os
from dataclasses import dataclass
import random

# TODO: for augmentation of triphone,
#  make sure the left and right of the current phone are not augmented?

N_PHONES = 44
SIMILAR_PHONES = [
    ['AA', 'AH'],
    ['AE', 'EH'],
    ['IY', 'IH'],
    ['UH', 'UW'],
]
ph2similar = {}
for group in SIMILAR_PHONES:
    for curr in group:
        ph2similar[curr] = [p for p in group if p != curr]


@dataclass(frozen=True)
class SourceLoc:
    utt: str
    idx: int  # position in sentence, used to reconstruct the sentence


@dataclass(frozen=True)
class Phone:
    name: str
    loc: SourceLoc
    score: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Phone transcripts')
    parser.add_argument('--scores', type=str, help='Path to utt2scores')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--rule-path', type=str, help='Path to rule.yaml file')
    return parser.parse_args()


def load_into_phones(text_path: str, score_path: str) -> List[Phone]:
    ret = []
    utt2phones = load_utt2seq(text_path)
    utt2scores = load_utt2seq(score_path, formatter=int)
    for utt in utt2phones.keys():
        phones = [p for p in utt2phones[utt] if p not in EMPTY_PHONES]
        scores = utt2scores[utt]
        assert len(scores) == len(phones)

        phones = [Phone(name=p, score=scores[i], loc=SourceLoc(utt=utt, idx=i)) for i, p in enumerate(phones)]
        ret += phones

    return ret


def samples_to_ph2samples(data: List[Phone]) -> Dict[str, List[Phone]]:
    ph2samples: Dict[str, List[Phone]] = {}
    for sam in data:
        phone = sam.name
        ph2samples.setdefault(phone, []).append(sam)

    return ph2samples


def remove_duplicated_samples(samples: List[Phone]) -> List[Phone]:
    return list(set(samples))


def add_more_negative_data(data: List[Phone], rule_path: str) -> List[Phone]:
    """
    Take the 2-score samples of other phones as new 0-score samples, and take the 2-score samples of
    other similar phones as new 1-score samples
    """
    import yaml
    rules = yaml.full_load(open(rule_path).read())

    ph2samples = samples_to_ph2samples(data)
    phone_names = list(set([d.name for d in data]))

    ret = []
    for ph, samples in ph2samples.items():
        # NOT including 1-score samples
        # invalid_candidates = [sam for sam in samples if sam.score != 2]
        # ret += invalid_candidates

        samples = [sam for sam in samples if sam.score == 2]
        random.shuffle(samples)
        # sam_0s, sam_1s, sam_2s = [samples[i::3] for i in range(3)]
        # assert len(sam_0s) + len(sam_1s) + len(sam_2s) == len(samples)
        sam_0s, sam_2s = [samples[i::2] for i in range(2)]
        assert len(sam_0s) + len(sam_2s) == len(samples)

        # 0-score samples
        other_phones = [p for p in phone_names if p in rules[ph]]
        for s in sam_0s:
            ret.append(Phone(name=random.choice(other_phones), loc=s.loc, score=0))

        # 1-score samples
        # similar_phones = [p for p in ph2similar.get(ph, [])]
        # if len(similar_phones) > 0:
        #     for s in sam_1s:
        #         ret.append(Phone(name=random.choice(similar_phones), loc=s.loc, score=1))
        # else:
        #     ret += sam_1s

        ret += sam_2s

    return ret


def restore_utt2(data: List[Phone]) -> (Dict[str, List[str]], Dict[str, List[int]]):
    utt2samples = {}
    for phone in data:
        utt2samples.setdefault(phone.loc.utt, []).append(phone)

    utt2phones = {}
    utt2scores = {}
    for utt, samples in utt2samples.items():
        samples = sorted(samples, key=lambda p: p.loc.idx)
        utt2phones[utt] = [s.name for s in samples]
        utt2scores[utt] = [s.score for s in samples]

    return utt2phones, utt2scores


def main():
    args = get_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data = load_into_phones(args.text, args.scores)
    new_data = add_more_negative_data(data, args.rule_path)
    data = new_data

    utt2phones, utt2scores = restore_utt2(data)

    score_counts = {i: 0 for i in range(3)}
    for d in data:
        score_counts[d.score] += 1
    print(f'Score counts after data aug: {score_counts}')

    utt2phones = dict(sorted(utt2phones.items(), key=lambda item: item[0]))
    utt2scores = dict(sorted(utt2scores.items(), key=lambda item: item[0]))

    write_utt2seq(os.path.join(output_dir, 'ref_aug.txt'), utt2phones)
    write_utt2seq(os.path.join(output_dir, 'utt2scores_aug'), utt2scores)


if __name__ == '__main__':
    main()
