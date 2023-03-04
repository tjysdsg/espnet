"""
Create text and utt2spk

Based on https://github.com/monirome/AphasiaBank/blob/main/clean_transcriptions.ipynb
"""
import os
import re
from argparse import ArgumentParser
from config import get_utt, pwa_spks, control_spks

import pylangacq as pla


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--transcript-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--story", type=str, required=True, help='Story name is used as the prefix of utt_id')
    parser.add_argument("--tag-insertion", type=str, choices=['none', 'prepend', 'append', 'both'])
    parser.add_argument("--include-investigators", action='store_true', help='Include investigator data')
    return parser.parse_args()


def clean_trans(trans: str) -> str:
    trans = trans.split("\x15")[0]  # remove timestamp
    trans = trans.lower()

    # Precodes and postcodes
    trans = re.sub(r"\[[+-] \w+]?", "", trans)

    # Word errors
    # '[: put] [* p:w-ret]' '[: princess]'
    trans = re.sub(r"\[: .+?] \[\* \S+]", "", trans)
    trans = re.sub(r"\[: .+?]", "", trans)  # [: x [* n:uk]
    trans = re.sub(r"\[\* .+?]", "", trans)  # [* s:r-ret]
    trans = re.sub(r"\[\*]", "", trans)  # [*]

    # Interruption by other people
    trans = re.sub(r"\*\w+:[\w_]+", "", trans)  # *INV: thank_you

    # Comments, explanation, paralinguistic
    trans = re.sub(
        r"\[% .+?]", "", trans
    )  # [% mimics opening her eyes for the first time]
    trans = re.sub(r"\[= .+?]", "", trans)  # [= growling voice]
    trans = re.sub(r"\[=! [\w :_]+?]", "", trans)  # [=! laughing]

    # Retracing and repetition
    trans = re.sub(r"\[/+]", "", trans)
    # we shouldn't remove this but there is only one occurrence in UNH03b
    trans = re.sub(r"\[x \d]", "", trans)

    # Pauses
    trans = re.sub(r"\([\d:.]+\)", "", trans)  # (.) (..) (2:13.12)

    # Special utterance terminators
    trans = re.sub(
        r"\+[<+\",./?]+", "", trans
    )  # +... +..? +/. +/? +//. +//? +, ++ +"/. +" +". +<

    # Overlap precedes
    trans = re.sub(r"\[[<>]\d?]", "", trans)  # [<]   [<1]  [>]  [>2]

    # Special form markers (CHAT manual page 45)
    trans = re.sub(r"@\S+", "", trans)

    # Local events and gestures
    trans = re.sub(r"[<>]", "", trans)  # remove all <> before adding <xxx>
    # laughter
    trans = re.sub(r"&=laughs", "<LAU>", trans)
    trans = re.sub(r"&=chuckles", "<LAU>", trans)
    # '&=points:picture_1'  '&=clears:throat'  '&=twiddles'
    trans = re.sub(r"&=\w*[:\w]*", "", trans)

    # Fillers and fragments
    trans = re.sub(r"&-", "", trans)  # &-uh  &-um
    trans = re.sub(r"&\+?", "", trans)  # phonological fragment

    # Special words
    trans = re.sub(r"www|xxx", "", trans)

    # Other remaining punctuations
    trans = re.sub(r"[,?!;:\"“%‘”.�‡„$^↓↑≠()\[\]↫\x15+]", "", trans)
    # '_', '-', and multiple whitespaces are replaced with a single whitespace
    trans = re.sub(r"_|-| {2,}", " ", trans)

    return trans.strip()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    utt_filter = None if args.include_investigators else 'PAR'

    # get a list of all CHAT files
    files = []
    for file in os.listdir(args.transcript_dir):
        if os.path.isfile(os.path.join(args.transcript_dir, file)) and file.endswith(".cha"):
            files.append(file)

    all_chars = set()
    with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as text:
        for file in files:
            spk = file.split(".cha")[0].replace('-', '')

            path = os.path.join(args.transcript_dir, file)
            chat: pla.Reader = pla.read_chat(path)

            utts = chat.utterances(participants=utt_filter)
            n_utts = len(utts)
            for i in range(n_utts):
                time_marks = utts[i].time_marks

                # skipping utterances without time marks
                if time_marks is None:
                    continue

                start, end = utts[i].time_marks

                tiers = utts[i].tiers
                is_investigator = False
                if "PAR" in tiers:
                    trans = clean_trans(tiers["PAR"])
                elif "INV" in tiers:
                    assert args.include_investigators
                    trans = clean_trans(tiers["INV"])
                    is_investigator = True
                else:
                    raise RuntimeError("Cannot find participant's or investigator's transcript")

                if len(trans) == 0:
                    continue

                # gather all unique chars
                for c in trans:
                    all_chars.add(c)

                # insert tags if needed
                if args.tag_insertion != "none":
                    if spk in pwa_spks:
                        aphasia_type = 'APH'
                    elif spk in control_spks:
                        aphasia_type = 'NONAPH'
                    else:
                        assert False

                    if args.tag_insertion == 'append':
                        trans = f"{trans} [{aphasia_type}]"
                    elif args.tag_insertion == 'prepend':
                        trans = f"[{aphasia_type}] {trans}"
                    elif args.tag_insertion == 'both':
                        trans = f"[{aphasia_type}] {trans} [{aphasia_type}]"
                    else:
                        assert False

                utt_id = get_utt(f'{spk}I' if is_investigator else spk, args.story, start, end)
                text.write(f"{utt_id}\t{trans}\n")

    print(all_chars)


if __name__ == "__main__":
    main()
