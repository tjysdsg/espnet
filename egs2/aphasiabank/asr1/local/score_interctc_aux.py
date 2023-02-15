import argparse
from config import utt2spk, spk2aphasia_label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    return parser.parse_args()


def main():
    args = get_args()

    utt_num = 0
    correct_aph = 0

    with open(args.hyp, encoding='utf-8') as f:
        for hyp in f:
            hyp = hyp.strip().split()  # utt [APH]

            utt_id = hyp[0]
            hyp_aph = hyp[1]

            spk = utt2spk(utt_id)
            ref_aph = spk2aphasia_label[spk]
            ref_aph = f"[{ref_aph.upper()}]"

            if ref_aph == hyp_aph:
                correct_aph += 1

            utt_num += 1

    print(
        f"Aphasia Detection: Accuracy "
        f"{(correct_aph / float(utt_num)):.4f} ({correct_aph}/{utt_num})\n"
    )


if __name__ == "__main__":
    main()
