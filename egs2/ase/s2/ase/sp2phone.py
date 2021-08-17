import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Calculate WER')
    parser.add_argument('--input', type=str, help='Hypothesis file')
    parser.add_argument('--output', type=str, help='Output directory')
    return parser.parse_args()


def remove_score(sp: str) -> str:
    if sp[-1].isdigit():
        sp = sp[:-1]
    return sp


def main():
    args = get_args()

    of = open(args.output, 'w')
    with open(args.input) as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0]
            phones = tokens[1:]
            phones = ' '.join([remove_score(p) for p in phones])
            of.write(f'{utt}\t{phones}\n')

    of.close()


if __name__ == '__main__':
    main()
