- In `asr.sh`, two new tokens are added, `[APH]` and `[NONAPH]`. They are also in [local/nlsyms.txt](local/nlsyms.txt)
  By doing this, the `CommonPreprocessor` and `TokenIDConverter` will treat these two as individual tokens instead of
  `'[', 'A', 'P', 'H', ']'`
- [score_cleaned.sh](score_cleaned.sh) is used to calculate CER per language per Aphasia. It doesn't require the input
  hypothesis file to contain language or Aph tags. But if the input does contain, it will automatically remove them
  before calculation
- [score_interctc_aux.sh](score_interctc_aux.sh) is used to calculate the Aphasia detection accuracy of the InterCTC
  layers. It can also score speaker-level aphasia detection accuracy.
- [local/score_lang_and_aphasia.py](local/score_lang_and_aphasia.py) is used to calculate language detection accuracy
  and the Aphasia detection accuracy in experiments that the hypothesis text file contains `[EN]` and `[APH]` tags.
  Pass in `--aph-field` to specify the location of relevant tags. For example `--aph-field=-1` if the Aph tags are at
  the end of every sentence
- [local/config.py](local/config.py) contains the information about every speaker and the tran/test/val split

# How to make the `[APH]` tag be treated as a single token

https://github.com/espnet/espnet/blob/master/espnet2/text/char_tokenizer.py#L45 char tokenizer treat everything as
length-1 string (character) except non-linguistic symbols, so I have to put `[APH]` and `[NONAPH]` into `nlsyms.txt`.

Since remove_non_linguistic_symbols is true during training, it gets kept in the returned array, something
like `[..., ..., '[APH]']`

Then I put `[APH]` and `[NONAPH]` into `tokens.txt`, then
https://github.com/espnet/espnet/blob/master/espnet2/text/token_id_converter.py#L35 token id converter will use it in
`tokens2ids()`

# Remove `--nlsyms_txt "local/nlsyms.txt"` when running stage 13 for tag-based detectors

# Use `local/dementia/clean_score_dir.py` and `local/dementia/score_interctc_aux.py` in `score_cleaned.sh` and `score_interctc_aux.sh` for Dementia experiments
