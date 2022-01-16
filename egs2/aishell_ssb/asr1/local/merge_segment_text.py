import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--out-file', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    utt2text = {}
    with open(args.text) as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0]
            text = tokens[1:]

            orig_utt, seg_i = utt.split('.')
            seg_i = int(seg_i)

            utt2text.setdefault(orig_utt, {})[seg_i] = text

    with open(args.out_file, 'w') as f:
        for utt, seg in utt2text.items():
            text_seg = []
            n = len(seg)
            for i in range(n):
                text_seg += seg[i]

            text = ' '.join(text_seg)
            f.write(f'{utt}\t{text}\n')


if __name__ == '__main__':
    main()
