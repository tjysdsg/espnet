from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--token-type", type=str, required=True, choices=['char', 'word'])
    return parser.parse_args()


def clean_line(line: str, tok: str):
    try:
        i = line.rindex(tok)
    except ValueError as e:
        print(f"{e}\n\nLine is: {line}")
        exit(1)

    assert 0 < i < len(line) - len(tok), f"Invalid line: {line}"
    return line[i + len(tok):]


def main():
    args = get_args()

    tok = '] <space> ' if args.token_type == 'char' else '] '
    with open(args.input, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as of:
        for line in f:
            of.write(clean_line(line, tok))


if __name__ == "__main__":
    main()
