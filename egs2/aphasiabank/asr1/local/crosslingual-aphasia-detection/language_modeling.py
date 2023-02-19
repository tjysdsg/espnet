import re
import argparse
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Language Modeling")

    parser.add_argument(
        "--languages",
        choices=["all", "en", "fr", "el"],
        default="all",
        help="Languages",
    )
    parser.add_argument(
        "--download-wikipedia",
        action="store_true",
        help="Download wikipedia",
    )
    parser.add_argument(
        "--fix-model",
        action="store_true",
        help="Fix language model",
    )
    parser.add_argument(
        "--fix-model-name",
        type=str,
        help="Path to language model name",
    )

    return parser.parse_args()


def create_wikipedia_dataset(language: str):
    if language == "en":
        language_name = "english"
        dataset = load_dataset("wikipedia", "20200501.en")
    elif language == "fr":
        language_name = "french"
        dataset = load_dataset("wikipedia", "20200501.fr")
    elif language == "el":
        language_name = "greek"
        dataset = load_dataset(
            "wikipedia", language="el", date="20220120", beam_runner="DirectRunner"
        )
    else:
        msg = "Improper language name. Process terminates."
        exit(msg)

    dataset = dataset["train"]
    chars_to_ignore_regex = '[,?.!\-\;\:"“%‘”�—’…–]'

    def extract_text(batch):
        text = batch["text"]
        batch["text"] = re.sub(chars_to_ignore_regex, "", text.lower())
        return batch

    dataset = dataset.map(extract_text)

    with open(f"text_{language_name}_wikipedia.txt", "w") as f:
        f.write(" ".join(dataset["text"]))


def fix_model(model_name: str):
    with open(f"{model_name}.arpa", "r") as read_file, open(
            f"{model_name}_correct.arpa", "w"
    ) as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count = line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count) + 1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)


if __name__ == "__main__":
    args = parse_arguments()
    if args.download_wikipedia:
        if args.languages == "all":
            a = list(map(create_wikipedia_dataset, ["en", "fr", "el"]))
        else:
            create_wikipedia_dataset(language=args.languages)

    if args.fix_model:
        fix_model(model_name=args.fix_model_name)
