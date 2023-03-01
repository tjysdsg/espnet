"""
Overwrite some of the content of text using sudo_text
"""
def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('text', type=str)
    parser.add_argument('sudo_text', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    utt2text = {}
    ordered_keys = []
    with open(args.text, encoding='utf-8') as f:
        for line in f:
            utt, value = line.rstrip('\n').split(maxsplit=1)
            utt2text[utt] = value
            ordered_keys.append(utt)

    utt2sudo_text = {}
    with open(args.sudo_text, encoding='utf-8') as f:
        for line in f:
            utt, value = line.rstrip('\n').split(maxsplit=1)
            utt2sudo_text[utt] = value

    # overwrite original text with sudo text
    for utt, sudo in utt2sudo_text.items():
        utt2text[utt] = sudo

    with open(args.output, 'w', encoding='utf-8') as f:
        for utt in ordered_keys:
            f.write(f'{utt} {utt2text[utt]}\n')


if __name__ == '__main__':
    main()
