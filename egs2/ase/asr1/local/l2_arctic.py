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
parser.add_argument("--l2_path", help="l2-Arctic path")
parser.add_argument("--save_path", help="l2-Arctic path")

args = parser.parse_args()

path = args.l2_path + "/*/annotation/*.TextGrid"
# Spanish\Vietnamese\Hindi\Mandarin\Korean\Arabic
train_spk = ["EBVS", "ERMS", "HQTV", "PNV", "ASI", "RRBI", "BWC", "LXC", "HJK", "HKK", "ABA", "SKA"]
dev_spk = ["MBMPS", "THV", "SVBI", "NCC", "YDCK", "YBAA"]
test_spk = ["NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"]
load_error_file = ["YDCK/annotation/arctic_a0209.TextGrid", "YDCK/annotation/arctic_a0272.TextGrid"]

wav_lst = glob.glob(path)
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
wrd_text = open(save_path + "/wrd_text", 'w')
wavscp = open(save_path + "/wav.scp", 'w')
ppl = open(save_path + "/phn_text", 'w')  # perceived phones
cpl = open(save_path + "/transcript_phn_text", 'w')  # correct phones


def del_repeat_sil(phn_lst):
    tmp = [phn_lst[0]]
    for i in range(1, len(phn_lst)):
        if phn_lst[i] == phn_lst[i - 1] and phn_lst[i] == "sil":
            continue
        else:
            tmp.append(phn_lst[i])
    return tmp


def clean_phone(phone: str):
    if phone == "sp" or phone == "SIL" or phone == " " or phone == "spn":
        ret = "sil"
    else:
        phone = phone.strip(" ")
        if phone == "ERR" or phone == "err":
            ret = "err"
        elif phone == "ER)":
            ret = "er"
        elif phone == "AX" or phone == "ax" or phone == "AH)":
            ret = "ah"
        elif phone == "V``":
            ret = "v"
        elif phone == "W`":
            ret = "w"
        else:
            ret = phone.lower()

    return ret.upper()  # FIXME


def main():
    for phn_path in wav_lst:
        phn_path = phn_path.replace('\\', '/')
        if "/".join(phn_path.split("/")[-3:]) in load_error_file:
            continue

        spk_id = phn_path.split("/")[-3]
        utt_id = spk_id + "-" + phn_path.split("/")[-1][:-9]
        tmp = re.sub("annotation", "wav", phn_path)
        wav_path = re.sub("TextGrid", "wav", tmp)
        tmp = re.sub("annotation", "transcript", phn_path)
        text_path = re.sub("TextGrid", "txt", tmp)

        cur_phns = []
        transcript_phns = []
        tg = textgrid.TextGrid.fromFile(phn_path)
        for i in tg[1]:
            if i.mark == '':
                transcript_phns.append("SIL")
                cur_phns.append("SIL")
            else:
                trans_human_type = i.mark.split(",")  # 'CPL,PPL,s'
                if len(trans_human_type) == 1:
                    phn = trans_human_type[0]
                else:
                    phn = trans_human_type[1]

                trans_phn = trans_human_type[0]
                trans_phn = trans_phn.rstrip(string.digits)

                # phn
                phn = phn.rstrip(string.digits + '*_')
                cur_phns.append(clean_phone(phn))

                # trans phn
                transcript_phns.append(clean_phone(trans_phn))

        f = open(text_path, 'r')
        for line in f:
            wrd_text.write(utt_id + " " + line.lower() + "\n")

        wavscp.write(utt_id + " " + wav_path + "\n")
        ppl.write(utt_id + " " + " ".join(del_repeat_sil(cur_phns)) + "\n")
        cpl.write(utt_id + " " + " ".join(del_repeat_sil(transcript_phns)) + "\n")

    wrd_text.close()
    wavscp.close()
    ppl.close()
    cpl.close()


if __name__ == '__main__':
    main()
