![](./images/logo.png)

:construction: The repo is under construction :construction:

This is the official source code for the paper "Zero-Shot Cross-lingual Aphasia Detection using Automatic Speech
Recognition" submitted to INTERSPEECH 2022.

## Installation

### General Package Installation

```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Download Dependency parser models

```
# Donwload English parser
python3 -m spacy download en_core_web_trf

# Download French parser
python3 -m spacy download fr_dep_news_trf

# Download Greek parser
python3 -m spacy download el_core_news_lg

```

## Datasets

- [AphasiaBank Dataset (English and French)](https://aphasia.talkbank.org/):
  "Researchers, educators, and clinicians working with aphasia who are interested in joining the consortium should read
  the Ground Rules and then send email to -EMAIL- with contact information and affiliation."
- Greek Corpus comes from [GREECAD](https://aphasia.talkbank.org/publications/2016/Varlokosta16.pdf) and
  the [PLan-V project](https://planv-project.gr/)

Since Greek data is not publicly available at the moment, we offer code and instructions mainly for English and French.

Currenly, we support the following folder structure for the transcripts dataset.

```
.
├── ...
├── French                          # French Folder
├── English                         # English Folder
│   ├── Aphasia                     # Aphasia Folder
│   ├── Control                     # Control Folder
│       ├── ...  
│       ├── [Institution Name]      # Institution Name folder
│       │   ├── [Speaker Name].cha  # speaker-level chat file
│       └── ...                        
└── ...
```

As for the root dir of audio files, we keep the [folder structure](https://media.talkbank.org/aphasia/) of AphasiaBank.
Since `mp4` files are not supported in our scripts, you need to convert them to `mp3` or `wav`.

## 1. Transcript Processing

To create initial data dictionary for a language in AphasiaBank run:

```
python3 transcript_processing.py --create-data-aphasiabank --transcripts-root-path {ROOT_PATH} --wavs-root-path {WAV_ROOT_PATH} --language fr --save-dict-name {SAVE_DICT}.json 
```

## 2. Dependency Parsing

To get the conllu output of every oracle utterance in the dictionary, run:

```
python3 dependency_parser.py --language {LANGUAGE} --data-dict-path {DATA_DICT}.json  --save-dict-name {SAVE_DICT}.json
```

If you want to calculate the conllu output of the ASR transcript add the  `--use-asr-transcripts` argument.

## 3. Feature Extraction

According to the paper, we calculate linguistic features that evaluate 6 language abilities, i.e., linguistic
productivity, content richness, fluency, syntactic complexity, lexical diversity, and gross output. These 24 features
are calculated at a speaker-level, and a complete list is presented in the table below.

| Language ability        | Feature                              | 
|-------------------------|--------------------------------------|
| Linguistic Productivity | Mean Length of Utterance             |  
| Content Richness        | Verb/Word Ratio                      |
|                         | Noun/Word Ratio                      |
|                         | Adjective/Word Ratio                 |
|                         | Adverb/Word Ratio                    |
|                         | Preposition/Word Ratio               |
|                         | Propositional density                |
| Fluency                 | Words per Minute                     |
| Syntactic Complexity    | Verbs per Utterance                  |
|                         | Noun Verb Ratio                      |
|                         | Open-closed class words              |
|                         | Conjunction/Word Ratio               |
|                         | Mean Clauses per utterance           |
|                         | Mean dependent clauses               |
|                         | Mean independent clauses             |
|                         | Dependent to all clauses ratio       |
|                         | Mean Tree height                     |
|                         | Max Tree depth                       |
|                         | Number of independent clauses        |
|                         | Number of dependent clauses          |
| Lexical Diversity       | Lemma/Token Ratio                    |
|                         | Words in Vocabulary per Words        |
|                         | Unique words in vocabulary per Words |
| Gross output            | Number of words                      |

### 3.1 Extract Features from dictionary

Currently, we support extracting features directly from dictionary at a speaker-level. To do so, run:

```
python3 feature_extraction.py --calculate-features-from-dict --data-dict-path {DATA_DICT}.json  --save-dict-name {SAVE_DICT}.json --language en 
```

If now, you follow the dictionary format we have proposed, you can calculate ASR features by adding the
argument `--use-asr-transcripts`.

## 4. Language Modeling

Language models intergrated in Automatic Speech Recognition architectures usually offer better WER results. In our case,
we train on Wikipedia a Kneser-Ney smoothed 3-gram language model using the KENLM toolkit for each language. Our
implementation is based
on [Hugging Face's google colab notebook](https://huggingface.co/blog/wav2vec2-with-ngram#:~:text=Boosting%20Wav2Vec2%20with%20n%2Dgrams%20in%20%F0%9F%A4%97%20Transformers,-Published%20January%2012&text=Wav2Vec2%20is%20a%20popular%20pre,for%20speech%20recognition%2C%20e.g.%20G)
🤗

### 4.1 Download Wikipedia dataset

You can can use several datasets to train your language models. For this work, we chose Wikipedia dataset, which serves
as a large generic out-of-domain dataset. Downloading Wikipedia dataset may take some time. To do so, run:

```
# To download Wikipedia dataset for every language
python3 language_modeling.py --download-wikipedia

# To download Wikipedia dataset for a specific language
# Supported languages: "en" (english), "fr" (french), "el" (greek)
python3 language_modeling.py --download-wikipedia --languages el
```

For each language, depending on your choice, this process outputs a txt file in the
format `text_{LANGUAGE}_wikipedia.txt`.

### 4.2 Setting up kenlm

Descriptive instructions about setting up `kenlm` can be found in the notebook mentioned in the beginning of this
section. Briefly, you need first to install some dependencies

```
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
```

and then, download the `kenlm` repo

```
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
```

### 4.3 Building a 3-gram model

For the language (`LANGUAGE`) of your choice, in order to create a 3-gram model, you can run the following command. Feel
free to explore the [`kenlm` documentation](https://kheafield.com/code/kenlm/) for more details.

```
kenlm/build/bin/lmplz -o 3 <"text_{LANGUAGE}_wikipedia.txt" > "3gram_{LANGUAGE}.arpa"
```

Then, following the instructions of the Google Colab notebook, we need to make some changes in the language models in
order not to face errors in the future.

```
python3 language_modeling.py --fix-model --fix-model-name 3g_{LANGUAGE}
```

After running this command, a `3g_{LANGUAGE}_correct.arpa` model will be created. These language models (especially the
English one) are large, and loading them may take some time. For this reason, you may need to create a binary version of
them. To do so, try:

```
build_binary -T /bigdisk 3g_{LANGUAGE}_correct.arpa 3g_{LANGUAGE}_correct.binary
```

## 5. Automatic Speech Recognition models

Our suggested ASR model uses the XLSR-53 architecture which is based on the Wav2Vec2.0 audio representation and was
trained with 56,000 hours of speech data in 53 languages. For each language, we use pretrained models from Hugging Face
that are finetuned in our languages i.e., English, French and Greek.

| Language | Model                                                                                                  | 
|----------|--------------------------------------------------------------------------------------------------------|
| English  | [wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) |
| French   | [wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)   | 
| Greek    | [wav2vec2-large-xlsr-53-greek](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-greek)     |   

### 5.1 Extracting logits from pretrained models

Because we explore various decoding techniques together with the use of language models, we first extract the logits
from every `audio file` and we store them in their corresponding folder for later use.

```
python3 asr.py --asr-logits-extraction --language en --asr-logits-save-path logits-english --data-dict-path {DATA_DICT}.json 
```

### 5.2 Decoding with or without a Language Model

Use this command to get ASR predictions without a Language models (greedy decoding).

```
python3 asr.py --asr-decoding --language {LANGUAGE} --asr-logits-path {LOGITS_LANGUAGE} --data-dict-path {DATA_DICT}.json --save-dict-name {SAVE_DICT} 
```

To run the ASR architecture with the ASR model, follow the command below. By default, we use `beam_width=10`.

```
python3 asr.py --asr-decoding --add-language-model --language {LANGUAGE} --asr-logits-path {LOGITS_LANGUAGE} --save-dict-name {SAVE_DICT} --kenlm-model-path {MODEL_PATH.{arpa/binary}}
```

### 5.3 ASR Evaluation

You can evaluate your ASR outputs with respect to Word Error Rate (WER) and Character Error Rate (CER) metrics by using
the following command. These metrics ara calculated both per label (i.e., aphasia and control) and in total.

```
python3 asr.py --asr-evaluation --data-dict-path {DATA_DICT}.json
```

## 6. Experiments

### 6.1 Monolingual experiment

To run the monolingual English Leave-One-Subject-Out (LOSO) experiment, execute the following command:

```
python3 experiments.py --mono-english --data-dict-path-source {DATA_DICT}.json 
```

### 6.2 Zero-shot Cross-lingual experiment

For the zero shot cross-lingual experiment, run the following command. Source corresponds to English data dict and
Target to French or Greek.

```
python3 experiments.py --zero-shot --data-dict-path-source {DATA_DICT_SOURCE_LANGUAGE}.json --data-dict-path-target {DATA_DICT_TARGET_LANGUAGE}.json --model {xgboost, svm, tree, forest}
```
