import regex
from typing import List

impure_pat = r'[_\d].*'
impure_matcher = regex.compile(impure_pat)

EMPTY_PHONES = ['sil', 'spn', 'eps']


def convert_to_pure_phones(phone: str) -> str:
    return impure_matcher.sub('', phone)


def remove_empty_phones(phones: List[str]) -> List[str]:
    return [p for p in phones if p.lower() not in EMPTY_PHONES]
