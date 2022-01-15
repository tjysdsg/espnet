import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--align-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'utt2align.json')) as f:
        utt2align = json.load(f)

    of_cache = {}
    for utt, align in utt2align.items():
        orig_utt = utt.split('.')[0]
        if orig_utt in of_cache:
            of = of_cache[orig_utt]
        else:
            of = open(os.path.join(out_dir, f'{orig_utt}_clean.txt'), 'w')
            of_cache[orig_utt] = of

        with open(os.path.join(args.align_dir, f'{utt}.txt')) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                _, _, start, end, _, phone = line.split()
                start, end = float(start), float(end)

                assert len(align) > 0

                # no need to optimize search since there are only a couple of segments per utt
                start_clean = 0
                for seg_start, seg_end, _ in align:
                    seg_dur = seg_end - seg_start
                    end_clean = start_clean + seg_dur

                    # FIXME: what if an aligned phone span across multiple segments?
                    if start_clean < start < end_clean:
                        break
                    start_clean += seg_end - seg_start
                start = start - start_clean + seg_start
                end = end - start_clean + seg_start

                of.write(f'{start:.2f} {end:.2f} {phone}\n')

    for of in of_cache.values():
        of.close()


if __name__ == '__main__':
    main()
