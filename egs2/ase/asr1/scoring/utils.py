import regex
from typing import List

impure_pat = r'[_\d].*'
impure_matcher = regex.compile(impure_pat)


def convert_to_pure_phones(phone: str) -> str:
    return impure_matcher.sub('', phone)


def remove_sil_from_phone_list(phones: List[str]) -> List[str]:
    return [p for p in phones if p.lower() != 'sil']
