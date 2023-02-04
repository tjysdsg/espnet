"""
Adapted from egs2/fleurs/asr1/local/score_lang_id.py
"""

import argparse
import codecs
import os
import sys
import traceback
from config import utt2spk, spk2lang_id, spk2aphasia_label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=str, help="folder of experiment", required=True
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The scoring output filename. " "If omitted, then output to sys.stdout",
    )
    return parser.parse_args()


def main():
    args = get_args()
    decode_folders = next(os.walk(args.exp))[1]
    for folder in decode_folders:
        if "decode_asr" in folder:
            scoring(args.exp, folder, args.out)


def scoring(exp_folder, decode_folder, out):
    exp_decode_folder = f"{exp_folder}/{decode_folder}"
    subfolders = next(os.walk(exp_decode_folder))[1]
    for folder in subfolders:
        decode_file_name = f"{exp_decode_folder}/{folder}/text"
        output_file = open(
            f"{exp_decode_folder}/{folder}/lang_id_refs.tsv", "w", encoding="utf-8"
        )
        output_file.write(f"utt_id\tref_lid\thyp_lid\n")

        try:
            decode_file = codecs.open(decode_file_name, "r", encoding="utf-8")
        except Exception:
            traceback.print_exc()
            print("\nUnable to open output file: " + decode_file_name)
            continue

        utt_num = 0
        correct_lid = 0
        correct_aph = 0

        while True:
            hyp = decode_file.readline()
            if not hyp:
                break

            hyp = hyp.strip().split()  # utt [EN] [APH]
            assert len(hyp) >= 3, hyp

            utt_id = hyp[0]
            hyp_lid = hyp[1]
            hyp_aph = hyp[2]

            spk = utt2spk(utt_id)
            ref_lid = spk2lang_id[spk]
            ref_lid = f"[{ref_lid.upper()}]"

            ref_aph = spk2aphasia_label[spk]
            ref_aph = f"[{ref_aph.upper()}]"

            if ref_lid == hyp_lid:
                correct_lid += 1
            if ref_aph == hyp_aph:
                correct_aph += 1
            utt_num += 1

            output_file.write(f"{utt_id}\t{ref_lid}\t{hyp_lid}\n")

        out.write(f"\n{exp_decode_folder}/{folder}\n")
        out.write(
            f"Language Identification Scoring: Accuracy "
            f"{(correct_lid / float(utt_num)):.4f} ({correct_lid}/{utt_num})\n"
        )
        out.write(
            f"Aphasia Detection Scoring: Accuracy "
            f"{(correct_aph / float(utt_num)):.4f} ({correct_aph}/{utt_num})\n"
        )


if __name__ == "__main__":
    main()
