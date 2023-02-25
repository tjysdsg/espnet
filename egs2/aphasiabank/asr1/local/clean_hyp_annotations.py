from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--tag-insertion", type=str, choices=['none', 'prepend', 'append', 'both'])
    return parser.parse_args()


# If aph tags are appended:
def clean_line_append(line: str) -> str:
    tok = "["

    try:
        i = line.rindex(tok)
        end_i = line.rindex(']')
    except ValueError as e:
        print(f"No language or aphasia tags found in:\n{line}")
        return line

    if end_i - i > 15:
        print(f'WARNING: too many characters are removed from: {line}')

    assert 0 < i < end_i, f"Invalid line: {line}"
    return line[:i] + line[end_i + 1:]


def clean_line_prepend(line: str) -> str:
    tok = "]"

    try:
        i = line.rindex(tok)
    except ValueError as e:
        print(f"No language or aphasia tags found in:\n{line}")
        return line

    if i + len(tok) > 15:
        print(f'WARNING: too many characters are removed from: {line}')
    assert 0 < i < len(line) - len(tok), f"Invalid line: {line}"
    return line[i + len(tok):]


def clean_line(line: str, tag_insertion: str) -> str:
    if tag_insertion == 'none':
        return line
    elif tag_insertion == 'append':
        return clean_line_append(line)
    elif tag_insertion == 'prepend':
        return clean_line_prepend(line)
    elif tag_insertion == 'both':
        return clean_line_append(clean_line_prepend(line))


def main():
    args = get_args()

    with open(args.input, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as of:
        for line in f:
            of.write(clean_line(line, args.tag_insertion))


if __name__ == "__main__":
    main()
