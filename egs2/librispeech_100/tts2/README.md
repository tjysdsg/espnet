Inference results of the 100 hour baseline on train_clean_360

```
2023-02-09T08:37:01 (asr.sh:1513:main) Write cer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/train_clean_360/score_cer/result.txt
|    SPKR      |    # Snt         # Wrd     |    Corr           Sub          Del           Ins          Err         S.Err    |
|    Sum/Avg   |   104014        19121267   |    98.2           1.1          0.8           0.6          2.5          78.8    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:38:20 (asr.sh:1513:main) Write wer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/train_clean_360/score_wer/result.txt
|    SPKR       |    # Snt        # Wrd     |    Corr          Sub           Del          Ins           Err        S.Err     |
|    Sum/Avg    |   104014       3595494    |    93.5          5.8           0.6          0.7           7.1         78.8     |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:38:34 (asr.sh:1513:main) Write cer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/test_clean/score_cer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2620        281530    |    98.2          1.1          0.7          0.6          2.5         59.8    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:38:38 (asr.sh:1513:main) Write wer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/test_clean/score_wer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2620         52576    |    93.4          6.0          0.6          0.7          7.3         59.8    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:38:51 (asr.sh:1513:main) Write cer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/test_other/score_cer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2939        272758    |    92.9          4.3          2.8          2.0          9.1         85.4    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:38:55 (asr.sh:1513:main) Write wer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/test_other/score_wer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2939         52343    |    81.0         16.8          2.1          1.9         20.8         85.4    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:39:09 (asr.sh:1513:main) Write cer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/dev_clean/score_cer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr         Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2703        288456    |    98.2         1.1          0.7          0.7          2.5         60.5    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:39:13 (asr.sh:1513:main) Write wer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/dev_clean/score_wer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr         Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2703         54402    |    93.4         6.0          0.6          0.7          7.2         60.5    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type char --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:39:25 (asr.sh:1513:main) Write cer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/dev_other/score_cer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr         Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2864        265951    |    92.9         4.4          2.7          2.1          9.2         84.4    |
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --cleaner none --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true
/jet/home/jtang1/miniconda3/bin/python3 /ocean/projects/cis210027p/jtang1/espnet/espnet2/bin/tokenize_text.py -f 2- --input - --output - --token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true --cleaner none
2023-02-09T08:39:29 (asr.sh:1513:main) Write wer result in exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/decode_asr_asr_model_valid.acc.ave/dev_other/score_wer/result.txt
|    SPKR      |    # Snt        # Wrd    |    Corr         Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg   |    2864         50948    |    81.5        16.6          2.0          1.9         20.4         84.4    |
```