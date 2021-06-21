import regex
from typing import List, Dict
import logging
import sys

impure_pat = r'[_\d].*'
impure_matcher = regex.compile(impure_pat)

EMPTY_PHONES = ['sil', 'spn', 'eps']


def convert_to_pure_phones(phone: str) -> str:
    return impure_matcher.sub('', phone)


def remove_empty_phones(phones: List[str]) -> List[str]:
    return [p for p in phones if p.lower() not in EMPTY_PHONES]


def remove_consecutive_phone(phones: List[str], val: str):
    ret = [phones[0]]
    n = len(phones)
    for i in range(1, n):
        if phones[i] == phones[i - 1] and phones[i] == val:
            continue
        else:
            ret.append(phones[i])
    return ret


def load_utt2phones(path: str, del_empty_phones=True) -> Dict[str, List[str]]:
    """
    Load utt2phones into a dictionary

    :return {utt: [phone1, phone2, ...]}
    """
    hyps = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0]
            hyp = tokens[1:]
            if del_empty_phones:
                phones = remove_empty_phones(hyp)
            hyps[utt] = phones

    return hyps


def load_so762_ref(text_phone: str) -> Dict[str, List[str]]:
    """
    Return a dictionary containing the correct phone transcripts of speechocean762 utterances

    :param text_phone The text-phone file in so762 dataset
    :return {utt: [phone1, phone2, ...]}
    """
    u2t2p = {}
    with open(text_phone) as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            assert len(tokens) > 1
            utt = tokens[0]
            trans = tokens[1:]

            assert '.' in utt
            utt, time = utt.split('.')
            time = int(time)
            if utt not in u2t2p:
                u2t2p[utt] = {time: trans}
            else:
                u2t2p[utt][time] = trans

    # sort phones by time
    u2p = {}
    for utt, t2p in u2t2p.items():
        temporal = sorted(t2p.items(), key=lambda x: x[0])
        phones = [p for _, ph in temporal for p in ph]
        u2p[utt] = [convert_to_pure_phones(p) for p in phones]

    return u2p


def create_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s: %(message)s', datefmt='%Y-%m-%d-%H-%M-%S')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger
