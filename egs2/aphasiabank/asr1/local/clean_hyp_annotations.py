import os
import re
import shutil
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('hyp_dir', type=str, help='Directory that contains the hypothesis "text" file')
    return parser.parse_args()


def main():
    args = get_args()
    hyp_dir = args.hyp_dir

    hyp_file = os.path.join(hyp_dir, 'text')

    # back up original text file
    backup_num = 0
    hyp_backup = os.path.join(hyp_dir, 'text_backup')
    while os.path.exists(hyp_backup):
        backup_num += 1
        hyp_backup = os.path.join(hyp_dir, f'text_backup.{backup_num}')

    shutil.copy2(hyp_file, hyp_backup)

    # remove all annotations from the hypothesis
    lines = []
    with open(hyp_file, encoding='utf-8') as f:
        for line in f:
            lines.append(re.sub(r"\[.*]\s", "", line))

    with open(hyp_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
