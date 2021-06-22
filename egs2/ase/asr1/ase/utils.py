import numpy as np
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


def onehot(n: int, idx: int) -> np.ndarray:
    ret = np.zeros(n, dtype='float32')
    ret[idx] = 1.0
    return ret
