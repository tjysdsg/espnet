"""
Based on https://github.com/cageyoko/CTC-Attention-Mispronunciation/blob/master/egs/attention_aug/local/l2arctic_prep.py
"""
import glob
import os
import string
import textgrid
import re
import argparse

parser = argparse.ArgumentParser(description="Prepare L2 data")
parser.add_argument("--l2-path", help="l2-Arctic path")
parser.add_argument("--output-dir", help="l2-Arctic path")

args = parser.parse_args()

path = args.l2_path + "/*/annotation/*.TextGrid"
train_spk = ["EBVS", "ERMS", "HQTV", "PNV", "ASI", "RRBI", "BWC", "LXC", "HJK", "HKK", "ABA", "SKA"]
dev_spk = ["MBMPS", "THV", "SVBI", "NCC", "YDCK", "YBAA"]
test_spk = ["NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"]

EMPTY_PHONES = [
    '<blank>',
    '<unk>',
    'SPN',
    'SIL',
    '<sos/eos>',
]

wav_lst = glob.glob(path)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
wrd_text = open(os.path.join(output_dir, "words"), 'w')
wavscp = open(os.path.join(output_dir, "wav.scp"), 'w')
ppl = open(os.path.join(output_dir, "ppl"), 'w')  # perceived phones
cpl = open(os.path.join(output_dir, "cpl"), 'w')  # correct phones
utt2spk = open(os.path.join(output_dir, "utt2spk"), 'w')


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


def main():
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
                cpl_phones.append("SIL")
                ppl_phones.append("SIL")
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

        # remove repeated SIL
        ppl_phones = del_repeat_sil(ppl_phones)
        cpl_phones = del_repeat_sil(cpl_phones)

        # for PPL, convert to score-phones
        ppl_phones = [phone_to_score_phone(p) for p in ppl_phones]

        f = open(text_path, 'r')
        for line in f:
            wrd_text.write(utt_id + " " + line.lower() + "\n")

        wavscp.write(f'{utt_id}\t{wav_path}\n')
        ppl.write(f'{utt_id}\t{" ".join(ppl_phones)}\n')
        cpl.write(f'{utt_id}\t{" ".join(cpl_phones)}\n')
        utt2spk.write(f'{utt_id}\t{spk_id}\n')

    wrd_text.close()
    wavscp.close()
    ppl.close()
    cpl.close()
    utt2spk.close()


if __name__ == '__main__':
    main()
