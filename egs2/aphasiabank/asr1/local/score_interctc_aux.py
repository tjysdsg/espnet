import argparse
from collections import Counter

from config import utt2spk, spk2aphasia_label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("--field", "-f", type=int, default=1,
                        help='The 0-based index of the field that is APH tag prediction')
    return parser.parse_args()


def main():
    args = get_args()

    utt_num = 0
    correct_aph = 0

    spk2tags = {}
    spk2ref = {}

    # sentence-level scoring
    with open(args.hyp, encoding='utf-8') as f:
        for hyp in f:
            hyp = hyp.strip().split()  # utt [APH]

            utt_id = hyp[0]
            hyp_aph = ''
            if len(hyp) > args.field:
                hyp_aph = hyp[args.field]
            else:
                print(f'WARNING: {spk} has no APH tag output')

            spk = utt2spk(utt_id)
            ref_aph = spk2aphasia_label[spk]
            ref_aph = f"[{ref_aph.upper()}]"
            spk2ref[spk] = ref_aph

            spk2tags.setdefault(spk, []).append(hyp_aph)

            if ref_aph == hyp_aph:
                correct_aph += 1

            utt_num += 1

    print('=' * 80)
    print(
        f"Sentence-level Aphasia detection accuracy "
        f"{(correct_aph / float(utt_num)):.4f} ({correct_aph}/{utt_num})"
    )
    print('=' * 80)

    # speaker-level scoring
    correct_speakers = 0
    n_speakers = 0
    for spk, tags in spk2tags.items():
        count = Counter(tags)
        n_aph = count['[APH]']
        n_nonaph = count['[NONAPH]']

        pred = '[NONAPH]'
        if n_aph >= n_nonaph:
            pred = '[APH]'

        if spk2ref[spk] == pred:
            correct_speakers += 1
        else:
            print(f'Incorrect majority voted prediction for {spk}: {count}')
        n_speakers += 1

    print('=' * 80)
    print(
        f"Speaker-level Aphasia detection accuracy "
        f"{(correct_speakers / float(n_speakers)):.4f} ({correct_speakers}/{n_speakers})"
    )
    print('=' * 80)


if __name__ == "__main__":
    main()
