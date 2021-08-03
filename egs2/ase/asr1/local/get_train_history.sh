#!/usr/bin/env bash

# run this under asr_train_*/
cat train.log | perl -lwne '/(\d+epoch).*(wer=[\d.]+)/i and print "$1 $2"'