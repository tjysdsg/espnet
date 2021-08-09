"""
Based on https://github.com/cageyoko/CTC-Attention-Mispronunciation/blob/master/egs/attention_aug/local/l2arctic_prep.py
"""
import glob
import os
import string
import textgrid
import re
import argparse
import shutil
from typing import List
from metrics import get_wer_details, predict_scores

EMPTY_PHONES = [
    '<blank>',
    '<unk>',
    'SPN',
    'SIL',
    '<sos/eos>',
]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare L2 data")
    parser.add_argument("--l2-path", help="l2-Arctic path")
    parser.add_argument("--output-dir", help="l2-Arctic path")
    return parser.parse_args()


def del_repeat_sil(phn_lst):
    tmp = [phn_lst[0]]
    for i in range(1, len(phn_lst)):
        if phn_lst[i] == phn_lst[i - 1] and phn_lst[i] == "SIL":
            continue
        else:
            tmp.append(phn_lst[i])
    return tmp


def clean_phone(phone: str):
    phone = phone.strip(" ").upper()
    if phone == "SP" or phone == "SIL" or phone == "" or phone == "SPN":
        ret = "SIL"
    else:
        if phone == "ER)":
            ret = "ER"
        elif phone == "AX" or phone == "AH)":
            ret = "AH"
        elif phone == "V``":
            ret = "V"
        elif phone == "W`":
            ret = "W"
        else:
            ret = phone

    return ret


def phone_to_score_phone(phone: str) -> str:
    if phone in EMPTY_PHONES:
        return phone

    if '*' in phone:
        return f'{phone.strip("*")}1'

    return f'{phone}2'


def get_scores(ppl: List[str], cpl: List[str]) -> List[int]:
    scores = predict_scores(['0'], get_wer_details({'0': ppl}, {'0': cpl}))
    return scores['0']


def clean_data(wav_lst: list, output_dir: str):
    wrd_text = open(os.path.join(output_dir, "words"), 'a')
    wavscp = open(os.path.join(output_dir, "wav.scp"), 'a')
    utt2scores = open(os.path.join(output_dir, "utt2scores"), 'a')
    ppl = open(os.path.join(output_dir, "text"), 'a')  # perceived phones
    cpl = open(os.path.join(output_dir, "cpl.txt"), 'a')  # correct phones
    utt2spk = open(os.path.join(output_dir, "utt2spk"), 'a')

    for phn_path in wav_lst:
        # PPL path
        phn_path = phn_path.replace('\\', '/')

        path_tokens = phn_path.split("/")

        # some files are broken
        load_error_file = ["YDCK/annotation/arctic_a0209.TextGrid", "YDCK/annotation/arctic_a0272.TextGrid"]
        if "/".join(path_tokens[-3:]) in load_error_file:
            continue

        spk_id = path_tokens[-3]

        # YDCK/annotation/arctic_a0209.TextGrid -> YDCK-arctic_a0209
        utt_id = spk_id + "-" + path_tokens[-1].split('.')[0]

        # wav path
        tmp = re.sub("annotation", "wav", phn_path)
        wav_path = re.sub("TextGrid", "wav", tmp)

        # CPL path
        tmp = re.sub("annotation", "transcript", phn_path)
        text_path = re.sub("TextGrid", "txt", tmp)

        ppl_phones = []
        cpl_phones = []
        tg = textgrid.TextGrid.fromFile(phn_path)
        for i in tg[1]:
            if i.mark == '':
                continue
            else:
                cpl_ppl_type = i.mark.split(",")  # [CPL] or [CPL, PPL, error_type]
                if len(cpl_ppl_type) == 1:  # no pronunciation error
                    ppl_phn = cpl_ppl_type[0]
                else:
                    ppl_phn = cpl_ppl_type[1]

                cpl_phn = cpl_ppl_type[0]

                # remove stress marker
                cpl_phn = cpl_phn.rstrip(string.digits)
                ppl_phn = ppl_phn.rstrip(string.digits)

                # clean phone
                ppl_phones.append(clean_phone(ppl_phn))
                cpl_phones.append(clean_phone(cpl_phn))

        # remove empty phones from CPL
        cpl_phones = [p for p in cpl_phones if p not in EMPTY_PHONES]

        # get scores
        scores = get_scores(ppl_phones, cpl_phones)  # NOTE: insertions in PPL are ignored

        # remove repeated SIL and convert to score-phones for PPL
        ppl_phones = del_repeat_sil(ppl_phones)
        ppl_phones = [phone_to_score_phone(p) for p in ppl_phones]

        f = open(text_path, 'r')
        for line in f:
            wrd_text.write(utt_id + " " + line.lower() + "\n")

        assert len(scores) == len(cpl_phones)

        wavscp.write(f'{utt_id}\t{wav_path}\n')
        utt2scores.write(f'{utt_id}\t{" ".join(map(str, scores))}\n')
        ppl.write(f'{utt_id}\t{" ".join(ppl_phones)}\n')
        cpl.write(f'{utt_id}\t{" ".join(cpl_phones)}\n')
        utt2spk.write(f'{utt_id}\t{spk_id}\n')

    wrd_text.close()
    wavscp.close()
    utt2scores.close()
    ppl.close()
    cpl.close()
    utt2spk.close()


def clean_data_for_spk(speakers: list, l2_path: str, output_dir: str):
    for spk in speakers:
        path = l2_path + f"/{spk}/annotation/*.TextGrid"
        wav_list = glob.glob(path)
        clean_data(wav_list, output_dir)


def main():
    args = get_args()
    train_spk = ["EBVS", "ERMS", "HQTV", "PNV", "ASI", "RRBI", "BWC", "LXC", "HJK", "HKK", "ABA", "SKA"]
    val_spk = ["MBMPS", "THV", "SVBI", "NCC", "YDCK", "YBAA"]
    test_spk = ["NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"]

    output_dir = os.path.join(args.output_dir, 'l2arctic_train')
    shutil.rmtree(output_dir, ignore_errors=True)  # remove previous data dir, since the files will be appended
    os.makedirs(output_dir)
    clean_data_for_spk(train_spk, args.l2_path, output_dir)

    output_dir = os.path.join(args.output_dir, 'l2arctic_test')
    shutil.rmtree(output_dir, ignore_errors=True)  # remove previous data dir, since the files will be appended
    os.makedirs(output_dir)
    clean_data_for_spk(test_spk, args.l2_path, output_dir)

    output_dir = os.path.join(args.output_dir, 'l2arctic_val')
    shutil.rmtree(output_dir, ignore_errors=True)  # remove previous data dir, since the files will be appended
    os.makedirs(output_dir)
    clean_data_for_spk(val_spk, args.l2_path, output_dir)


if __name__ == '__main__':
    main()
