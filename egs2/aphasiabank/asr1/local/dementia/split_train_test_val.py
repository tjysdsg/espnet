import os
from argparse import ArgumentParser
from config import train_spks, test_spks, utt2spk, utt2story


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, help="Path to text", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--include-investigators", action='store_true', help='Include investigator data')
    return parser.parse_args()


# NOTE: test set uses test_spks's cookie story segments, while validation set uses their other story sections
spk_splits = [train_spks, test_spks, test_spks]


def main():
    args = get_args()

    # get all speakers
    spk2utts = {}
    utt2trans = {}
    with open(args.text, encoding="utf-8") as f:
        for line in f:
            utt, trans = line.rstrip("\n").split(maxsplit=1)
            spk = utt2spk(utt)
            spk2utts.setdefault(spk, []).append(utt)
            utt2trans[utt] = trans

    splits = ["train", "val", "test"]
    out_dir = args.out_dir

    utt_splits = [[] for _ in splits]
    for i, split in enumerate(splits):
        subset_dir = os.path.join(out_dir, split)
        os.makedirs(subset_dir, exist_ok=True)

        utt_list = open(os.path.join(subset_dir, "utt.list"), "w", encoding="utf-8")
        text = open(os.path.join(subset_dir, "text"), "w", encoding="utf-8")
        utt2spk_file = open(os.path.join(subset_dir, "utt2spk"), "w", encoding="utf-8")

        spks = spk_splits[i]
        investigators_spks = []
        if args.include_investigators:
            investigators_spks = [f'{s}I' for s in spks]

        for spk in spks + investigators_spks:
            if spk not in spk2utts:
                print(
                    f"Skipping utterances of {spk} "
                    f"since they are not found in {args.text}"
                )
                continue

            for utt in spk2utts[spk]:
                story = utt2story(utt)

                # test data contains only the 'cookie' story sections from the test speakers
                # so that it matches ADReSS2020 test data
                if split == 'test' and story != 'cookie':
                    continue
                if split == 'test' and "I" in spk:
                    continue

                # val data contains:
                # - all investigators from the test speakers
                # - story sections other than 'cookie' from the test speakers
                if split == 'val' and "I" not in spk and story == 'cookie':
                    continue

                utt_splits[i].append(utt)

                utt_list.write(f"{utt}\n")
                text.write(f"{utt}\t{utt2trans[utt]}\n")
                utt2spk_file.write(f"{utt}\t{spk}\n")

        utt_list.close()
        text.close()
        utt2spk_file.close()

    # make sure there's no overlapping between splits
    utts1 = set(utt_splits[0])
    utts2 = set(utt_splits[1])
    utts3 = set(utt_splits[2])

    assert not bool(utts1 & utts2)
    assert not bool(utts1 & utts3)
    assert not bool(utts2 & utts3)


if __name__ == "__main__":
    main()
