from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--appended-aph", action='store_true')
    parser.add_argument(
        "--token-type", type=str, required=True, choices=["char", "word"]
    )
    return parser.parse_args()


# If aph tags are appended:
def clean_line_append(line: str, tok: str):
    try:
        i = line.rindex(tok)
        end_i = line.rindex(']')
    except ValueError as e:
        print(f"No language or aphasia tags found in:\n{line}")
        return line

    assert 0 < i < end_i, f"Invalid line: {line}"
    return line[:i] + line[end_i + 1:]


def clean_line(line: str, tok: str):
    try:
        i = line.rindex(tok)
    except ValueError as e:
        print(f"No language or aphasia tags found in:\n{line}")
        return line

    assert 0 < i < len(line) - len(tok), f"Invalid line: {line}"
    return line[i + len(tok):]


def main():
    args = get_args()

    if args.appended_aph:
        tok = "<space> [" if args.token_type == "char" else "["
    else:
        tok = "] <space> " if args.token_type == "char" else "] "
    with open(args.input, encoding="utf-8") as f, \
            open(args.output, "w", encoding="utf-8") as of:
        for line in f:
            if args.appended_aph:
                of.write(clean_line_append(line, tok))
            else:
                of.write(clean_line(line, tok))


if __name__ == "__main__":
    main()
