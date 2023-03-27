"""
All configurations of AphasiaBank data in one place
"""

# ======================
#      Data splits
#   train  val  test
#    56%   19%  25%
# ======================


train_english_pwa = [
    # no severity info: 31
    "cmu01a",
    "cmu01b",
    "cmu02a",
    "BU11a",
    "MMA04a",
    "aprocsa1554a",
    "aprocsa1713a",
    "aprocsa1731a",
    "kurland01a",
    "kurland01b",
    "kurland01e",
    "kurland01f",
    "kurland02f",
    "kurland02g",
    "kurland07c",
    "kurland07d",
    "kurland08c",
    "kurland09c",
    "kurland09d",
    "kurland12c",
    "kurland12d",
    "kurland13c",
    "kurland13d",
    "kurland15d",
    "kurland21d",
    "kurland21e",
    "kurland22c",
    "kurland15e",
    "kurland16c",
    "kurland16d",
    "kurland17d",
    # mild: 116
    "MMA20a",
    "UNH03b",
    "UNH03c",
    "kurland12a",
    "scale05c",
    "kurland05a",
    "scale17a",
    "fridriksson02a",
    "kurland09b",
    "tap04a",
    "adler25a",
    "elman15a",
    "MMA12a",
    "MSU02b",
    "kurland04b",
    "kurland07b",
    "elman01a",
    "kurland02c",
    "kurland03b",
    "tucson16a",
    "wozniak05a",
    "fridriksson09a",
    "williamson17a",
    "adler20a",
    "fridriksson05a",
    "ACWT04a",
    "adler14a",
    "williamson15a",
    "wozniak04a",
    "kurland04a",
    "scale35a",
    "scale21a",
    "MSU02a",
    "star03a",
    "fridriksson07a",
    "whiteside18a",
    "tucson06a",
    "kansas19a",
    "adler01a",
    "kurland08a",
    "kurland14a",
    "scale06b",
    "kurland17b",
    "MMA05a",
    "wozniak01a",
    "UNH03a",
    "kansas15a",
    "williamson09a",
    "tcu10b",
    "ACWT07a",
    "BU01a",
    "adler24a",
    "adler07a",
    "williamson08a",
    "MMA09a",
    "scale12b",
    "kansas11a",
    "tucson06b",
    "kurland10b",
    "kurland02d",
    "tap01a",
    "adler08a",
    "UNH01a",
    "kurland14b",
    "kurland25a",
    "kurland23c",
    "BU05a",
    "whiteside06a",
    "scale06a",
    "fridriksson01a",
    "elman05a",
    "MMA02a",
    "UNH11a",
    "elman04a",
    "MMA03a",
    "elman13a",
    "williamson05a",
    "whiteside09a",
    "kansas21a",
    "UNH01c",
    "scale06c",
    "UNH05a",
    "kempler02a",
    "scale22a",
    "kansas04a",
    "tucson18a",
    "elman10a",
    "whiteside20a",
    "BU04a",
    "kurland12b",
    "tucson10a",
    "williamson14a",
    "whiteside17a",
    "wright202a",
    "tcu01a",
    "whiteside19a",
    "cmu03a",
    "williamson18a",
    "scale34a",
    "kurland08b",
    "kurland17c",
    "thompson02a",
    "scale18d",
    "fridriksson09b",
    "fridriksson04a",
    "tucson20a",
    "kurland26b",
    "williamson10a",
    "wozniak06a",
    "williamson07a",
    "kurland28c",
    "whiteside13a",
    "MSU01a",
    "thompson14a",
    "adler12a",
    "thompson01a",
    # moderate: 83
    "MSU08b",
    "ACWT05a",
    "whiteside02a",
    "UNH08a",
    "kempler04a",
    "fridriksson10b",
    "thompson03a",
    "tcu03a",
    "tucson15b",
    "UNH10a",
    "williamson03a",
    "kansas22a",
    "scale26a",
    "cmu02b",
    "fridriksson06b",
    "BU09a",
    "garrett01a",
    "scale15b",
    "williamson16a",
    "whiteside10a",
    "ACWT02a",
    "scale10a",
    "tap02a",
    "scale23a",
    "adler10a",
    "fridriksson10a",
    "wright201a",
    "adler04a",
    "fridriksson12a",
    "tucson07a",
    "kurland01c",
    "tucson11a",
    "scale11a",
    "wright206a",
    "scale11b",
    "scale15c",
    "williamson19a",
    "kurland29a",
    "adler13a",
    "scale18c",
    "kurland29c",
    "scale13a",
    "scale30b",
    "kurland24b",
    "adler18a",
    "williamson01a",
    "adler02a",
    "kurland02b",
    "kurland20a",
    "thompson05a",
    "wozniak07a",
    "scale36a",
    "adler05a",
    "kansas20a",
    "MSU05a",
    "tucson08a",
    "wozniak03a",
    "tucson22a",
    "williamson06a",
    "MSU07a",
    "scale19a",
    "MMA16a",
    "BU07a",
    "whiteside12a",
    "whiteside08a",
    "scale05b",
    "kurland01d",
    "scale18a",
    "MSU03b",
    "BU02a",
    "scale04a",
    "tap14a",
    "kansas17a",
    "MMA13a",
    "adler15a",
    "tucson09a",
    "kansas09a",
    "williamson24a",
    "MSU07b",
    "scale12a",
    "thompson04a",
    "kansas13a",
    "kurland19b",
    # severe: 26
    "UNH04b",
    "scale25a",
    "BU08a",
    "whiteside03a",
    "kansas05a",
    "scale28a",
    "UNH04a",
    "tcu02b",
    "fridriksson03a",
    "MMA14a",
    "adler06a",
    "kansas12a",
    "ACWT11a",
    "tucson14a",
    "scale24a",
    "tucson15a",
    "tap13a",
    "kurland22b",
    "whiteside04a",
    "ACWT08a",
    "tucson03a",
    "elman06a",
    "elman08a",
    "scale27a",
    "fridriksson06a",
    "adler23a",
    # very severe: 9
    "kurland15c",
    "kansas01a",
    "scale09a",
    "kansas08a",
    "tap09a",
    "kansas02a",
    "kansas06a",
    "williamson21a",
    "adler19a",
]

train_french_pwa = [
    "lemeur09a",
    "lemeur11a",
    "colin08a",
    "colin09a",
    "colin10a",
    "colin10a2",
    "colin16a",
]

train_english_control = [
    # control: 151
    "richardson41",
    "MSUC09b",
    "richardson42",
    "capilouto45a",
    "capilouto32a",
    "capilouto20a",
    "wright71a",
    "UMD22",
    "capilouto60a",
    "capilouto47a",
    "capilouto26a",
    "capilouto61a",
    "wright94a",
    "capilouto68a",
    "richardson173",
    "capilouto23a",
    "wright70a",
    "capilouto67a",
    "wright90a",
    "capilouto41a",
    "richardson205",
    "capilouto53a",
    "wright92a",
    "wright97a",
    "wright91a",
    "capilouto04a",
    "capilouto24a",
    "richardson201",
    "richardson54",
    "wright95a",
    "wright26a",
    "richardson188",
    "UMD14",
    "wright57a",
    "richardson170",
    "capilouto59a",
    "wright09a",
    "richardson191",
    "UMD11",
    "MSUC07b",
    "capilouto28a",
    "capilouto78a",
    "MSUC08b",
    "capilouto06a",
    "capilouto79a",
    "UMD01",
    "richardson59",
    "wright58a",
    "wright82a",
    "wright24a",
    "wright05a",
    "capilouto42a",
    "richardson196",
    "wright33a",
    "capilouto33a",
    "wright55a",
    "wright102a",
    "richardson39",
    "richardson25",
    "wright45a",
    "capilouto35a",
    "capilouto14a",
    "capilouto49a",
    "wright74a",
    "capilouto17a",
    "wright85a",
    "capilouto51a",
    "wright61a",
    "capilouto11a",
    "capilouto18a",
    "wright30a",
    "MSUC02b",
    "UMD24",
    "capilouto16a",
    "UMD03",
    "wright72a",
    "MSUC04b",
    "wright96a",
    "wright77a",
    "wright86a",
    "wright49a",
    "wright93a",
    "capilouto40a",
    "richardson175",
    "wright67a",
    "wright73a",
    "capilouto65a",
    "richardson184",
    "capilouto07a",
    "wright12a",
    "richardson204",
    "wright62a",
    "wright42a",
    "wright02a",
    "richardson195",
    "capilouto80a",
    "wright99a",
    "wright19a",
    "capilouto34a",
    "richardson167",
    "richardson24",
    "wright87a",
    "richardson20",
    "capilouto30a",
    "capilouto25a",
    "UMD19",
    "capilouto50a",
    "capilouto15a",
    "richardson202",
    "UMD15",
    "wright39a",
    "wright64a",
    "richardson192",
    "richardson178",
    "capilouto10a",
    "richardson171",
    "capilouto43a",
    "richardson36",
    "richardson189",
    "wright83a",
    "capilouto62a",
    "richardson203",
    "richardson185",
    "wright79a",
    "wright53a",
    "MSUC06a",
    "wright101a",
    "capilouto77a",
    "UMD12",
    "wright32a",
    "capilouto58a",
    "wright100a",
    "wright48a",
    "wright36a",
    "richardson198",
    "wright66a",
    "richardson22",
    "richardson165",
    "wright07a",
    "wright22a",
    "capilouto54a",
    "MSUC09a",
    "wright28a",
    "richardson38",
    "richardson168",
    "capilouto09a",
    "wright63a",
    "wright43a",
    "richardson21",
    "richardson23",
    "wright11a",
]

train_french_control = [
    # French Control: 14
    "colin01a",
    "colin03a",
    "lemeur01a",
    "lemeur02a",
    "lemeur04a",
    "colin04a",
    "colin05a",
    "lemeur05a",
]

val_english_pwa = [
    # no severity info: 11
    "kurland17e",
    "kurland18c",
    "kurland18d",
    "kurland19d",
    "kurland19e",
    "aprocsa1738a",
    "aprocsa1833a",
    "aprocsa1944a",
    "kurland22d",
    "kurland22e",
    # mild: 39
    "UNH05b",
    "adler09a",
    "kurland21c",
    "kurland28a",
    "MSU06b",
    "kansas18a",
    "adler17a",
    "kansas07a",
    "MSU01b",
    "kurland03a",
    "scale20a",
    "whiteside01a",
    "scale17c",
    "MMA22a",
    "adler21a",
    "kurland06b",
    "kurland17a",
    "kurland25b",
    "UNH06a",
    "williamson13a",
    "kurland07a",
    "MBA01a",
    "MMA19a",
    "UCL04a",
    "wright203a",
    "tucson08b",
    "scale02b",
    "scale32a",
    "kansas03a",
    "whiteside05a",
    "scale06d",
    "ACWT09a",
    "tcu05a",
    "tap07a",
    "BU12a",
    "thompson06a",
    "scale16a",
    "tucson04a",
    "kurland21b",
    # moderate: 29
    "UNH07b",
    "whiteside14a",
    "tap10a",
    "MSU08a",
    "MMA15a",
    "tap15a",
    "scale30a",
    "UNH09a",
    "kurland24c",
    "kurland24a",
    "kurland27b",
    "UCL02a",
    "elman12a",
    "kurland19a",
    "wright207a",
    "tap17a",
    "tcu08a",
    "williamson04a",
    "tap11a",
    "ACWT10a",
    "BU10a",
    "elman07a",
    "kurland29b",
    "UCL01a",
    "wright205a",
    "kempler03a",
    "ACWT03a",
    "scale33a",
    "kurland27a",
    # severe: 10
    "UNH09b",
    "kurland18b",
    "UCL03a",
    "kurland22a",
    "kurland16b",
    "williamson23a",
    "UNH02b",
    "MMA08a",
    "kansas16a",
    "kurland18a",
    # very severe: 3
    "MMA10a",
    "kurland15a",
    "scale07a",
]

val_french_pwa = [
    "lemeur10a",
    "colin11a",
    "colin12a",
]

val_english_control = [
    # 50
    "richardson194",
    "capilouto08a",
    "richardson197",
    "capilouto27a",
    "wright15a",
    "capilouto55a",
    "richardson92",
    "UMD09",
    "wright29a",
    "richardson200",
    "capilouto12a",
    "capilouto03a",
    "wright52a",
    "richardson18",
    "MSUC03b",
    "UMD10",
    "richardson169",
    "wright38a",
    "wright16a",
    "MSUC05b",
    "wright04a",
    "wright21a",
    "capilouto56a",
    "richardson177",
    "wright47a",
    "capilouto48a",
    "UMD21",
    "wright59a",
    "wright50a",
    "UMD02",
    "wright80a",
    "wright68a",
    "MSUC04a",
    "wright69a",
    "richardson37",
    "wright20a",
    "capilouto01a",
    "UMD18",
    "wright46a",
    "MSUC01b",
    "wright89a",
    "richardson58",
    "wright88a",
    "wright98a",
    "capilouto44a",
    "richardson34",
    "wright35a",
    "UMD20",
    "capilouto66a",
    "capilouto05a",
]

val_french_control = [
    "colin02a",
    "lemeur03a",
]

test_english_pwa = [
    # no severity info: 15
    "kurland23d",
    "kurland23e",
    "kurland24d",
    "kurland24e",
    "kurland25d",
    "kurland25e",
    "kurland26d",
    "kurland26e",
    "kurland27d",
    "kurland27e",
    "kurland28d",
    "kurland29d",
    "kurland29e",
    "tucson17a",
    "williamson22a",
    # mild: 52
    "UNH05c",
    "ACWT12a",
    "MSU06a",
    "adler22a",
    "elman11b",
    "UNH01b",
    "tucson01a",
    "elman01b",
    "kurland26a",
    "williamson02a",
    "tcu09a",
    "MSU04b",
    "thompson07c",
    "scale12c",
    "BU06a",
    "whiteside07a",
    "kurland10a",
    "tap12a",
    "kurland26c",
    "wright204a",
    "fridriksson11a",
    "kurland05b",
    "MSU04a",
    "tap05a",
    "kurland02e",
    "MMA11a",
    "thompson11a",
    "kurland23a",
    "williamson14b",
    "thompson07a",
    "thompson13a",
    "adler03a",
    "kurland06a",
    "thompson10a",
    "kurland28b",
    "williamson09b",
    "kurland21a",
    "tap18a",
    "MMA17a",
    "BU03a",
    "thompson09a",
    "scale15d",
    "thompson07b",
    "wozniak02a",
    "kurland09a",
    "kurland23b",
    "scale08a",
    "thompson12a",
    "kurland25c",
    "garrett02a",
    "thompson08a",
    "scale12d",
    # moderate: 37
    "elman02a",
    "fridriksson13a",
    "MSU03a",
    "kurland13b",
    "ACWT01a",
    "tap19a",
    "scale15a",
    "scale18b",
    "elman11a",
    "scale01a",
    "kansas14a",
    "scale31a",
    "scale02a",
    "scale05a",
    "williamson11a",
    "kurland13a",
    "elman03a",
    "whiteside15a",
    "williamson12a",
    "scale14c",
    "elman14a",
    "kurland02a",
    "adler16a",
    "tap16a",
    "kurland19c",
    "whiteside16a",
    "tucson13a",
    "williamson12b",
    "elman09a",
    "kurland27c",
    "kansas10a",
    "tap08a",
    "scale14a",
    "whiteside11a",
    "MSU05b",
    "kansas23a",
    "scale38a",
    # severe: 12
    "tucson19a",
    "kurland16a",
    "tcu07a",
    "UNH07a",
    "tap06a",
    "tucson12a",
    "scale03a",
    "MMA06a",
    "tap03a",
    "fridriksson03b",
    "tucson02a",
    "UNH09c",
    # very severe: 4
    "adler11a",
    "fridriksson08b",
    "kurland15b",
    "UNH02a",
]

test_french_pwa = [
    "colin13a",
    "colin14a",
    "colin15a",
]

test_english_control = [
    # 67
    "richardson199",
    "wright03a",
    "capilouto29a",
    "MSUC02a",
    "wright31a",
    "UMD05",
    "wright13a",
    "richardson172",
    "capilouto31a",
    "wright81a",
    "richardson176",
    "UMD23",
    "wright60a",
    "richardson17",
    "MSUC03a",
    "capilouto52a",
    "wright37a",
    "richardson179",
    "MSUC06b",
    "capilouto57a",
    "capilouto19a",
    "wright10a",
    "wright14a",
    "richardson166",
    "richardson19",
    "wright34a",
    "capilouto39a",
    "wright51a",
    "MSUC01a",
    "richardson206",
    "MSUC05a",
    "wright17a",
    "UMD08",
    "capilouto13a",
    "capilouto64a",
    "wright08a",
    "richardson174",
    "wright78a",
    "UMD17",
    "richardson60",
    "wright27a",
    "wright65a",
    "wright84a",
    "MSUC08a",
    "wright23a",
    "UMD04",
    "wright01a",
    "capilouto22a",
    "capilouto46a",
    "richardson35",
    "capilouto37a",
    "UMD16",
    "wright06a",
    "capilouto02a",
    "wright75a",
    "capilouto36a",
    "wright18a",
    "wright25a",
    "capilouto21a",
    "UMD13",
    "UMD06",
    "wright40a",
    "capilouto63a",
    "richardson186",
    "kempler01a",
    "capilouto38a",
    "MSUC07a",
]

test_french_control = [
    "colin06a",
    "colin07a",
    "lemeur06a",
    "lemeur07a",
]

train_spks = (
        train_english_pwa + train_english_control + train_french_pwa + train_french_control
)
val_spks = val_english_pwa + val_english_control + val_french_pwa + val_french_control
test_spks = (
        test_english_pwa + test_english_control + test_french_pwa + test_french_control
)

assert len(train_spks) == len(set(train_spks)), f"Duplicated speakers in train"
assert len(val_spks) == len(set(val_spks)), f"Duplicated speakers in validation"
assert len(test_spks) == len(set(test_spks)), f"Duplicated speakers in test"

# make sure there is no overlap
assert not bool(set(train_spks) & set(test_spks)), set(train_spks) & set(test_spks)
assert not bool(set(train_spks) & set(val_spks)), set(train_spks) & set(val_spks)
assert not bool(set(test_spks) & set(val_spks)), set(test_spks) & set(val_spks)

all_spks = train_spks + val_spks + test_spks
english_spks = (
        train_english_pwa
        + train_english_control
        + val_english_pwa
        + val_english_control
        + test_english_pwa
        + test_english_control
)
french_spks = (
        train_french_pwa
        + train_french_control
        + val_french_pwa
        + val_french_control
        + test_french_pwa
        + test_french_control
)
pwa_spks = (
        train_english_pwa
        + train_french_pwa
        + val_english_pwa
        + val_french_pwa
        + test_english_pwa
        + test_french_pwa
)
control_spks = (
        train_english_control
        + train_french_control
        + val_english_control
        + val_french_control
        + test_english_control
        + test_french_control
)

# ======================
#        Mappings
# ======================


# aphasia_type2label = {
#     "": "NONAPH",
#     "control": "NONAPH",
#     "notaphasicbywab": "APH",
#     "conduction": "APH",
#     "anomic": "APH",
#     "global": "APH",
#     "transsensory": "APH",
#     "transmotor": "APH",
#     "broca": "APH",
#     "aphasia": "APH",
#     "wernicke": "APH",
# }

lang2label = {
    "English": "EN",
    "French": "FR",
}

# these speakers' aphasia type information is missing, but we actually know for sure if they have Aphasia or not
spk2missing_aphasia_types = {
    "UMD01": "control",
    "UMD02": "control",
    "UMD03": "control",
    "UMD04": "control",
    "UMD05": "control",
    "UMD06": "control",
    "UMD08": "control",
    "UMD09": "control",
    "UMD10": "control",
    "UMD11": "control",
    "UMD12": "control",
    "UMD13": "control",
    "UMD14": "control",
    "UMD15": "control",
    "UMD16": "control",
    "UMD17": "control",
    "UMD18": "control",
    "UMD19": "control",
    "UMD20": "control",
    "UMD21": "control",
    "UMD22": "control",
    "UMD23": "control",
    "UMD24": "control",
    "MSUC09b": "control",
    "richardson21": "control",
    "richardson22": "control",
    "richardson23": "control",
    "richardson34": "control",
    "richardson35": "control",
    "richardson36": "control",
    "richardson37": "control",
    "richardson38": "control",
    "richardson39": "control",
    "aprocsa1554a": "aphasia",
    "aprocsa1713a": "aphasia",
    "aprocsa1731a": "aphasia",
    "aprocsa1738a": "aphasia",
    "aprocsa1833a": "aphasia",
    "aprocsa1944a": "aphasia",
    "kurland24e": "aphasia",
    "kurland25e": "aphasia",
    "MMA04a": "aphasia",
    "MMA20a": "aphasia",
    "colin08a": "aphasia",
    "colin10a2": "aphasia",
}

spk2lang_id = {}
for spk in english_spks:
    spk2lang_id[spk] = "EN"
for spk in french_spks:
    spk2lang_id[spk] = "FR"

lang_id2spks = {}
for spk, lid in spk2lang_id.items():
    lang_id2spks.setdefault(lid, []).append(spk)

spk2aphasia_label = {}
for spk in pwa_spks:
    spk2aphasia_label[spk] = "APH"
for spk in control_spks:
    spk2aphasia_label[spk] = "NONAPH"

aph2spks = {}
for spk, aph in spk2aphasia_label.items():
    aph2spks.setdefault(aph, []).append(spk)

# speaker to severity level
# spk2severity = {}
# with open('local/spk_info.txt', encoding='utf-8') as f:
#     for line in f:
#         spk, _, severity, aph_type = line.rstrip('\n').split()
#         if aph_type == 'control':
#             continue
#         elif severity != 'none':
#             spk2severity[spk] = severity
# print(spk2severity)
spk2severity = {
    'ACWT01a': 'moderate', 'ACWT02a': 'moderate', 'ACWT03a': 'moderate', 'ACWT04a': 'mild',
    'ACWT05a': 'moderate', 'ACWT07a': 'mild', 'ACWT08a': 'severe', 'ACWT09a': 'mild', 'ACWT10a': 'moderate',
    'ACWT11a': 'severe', 'ACWT12a': 'mild', 'BU01a': 'mild', 'BU02a': 'moderate', 'BU03a': 'mild',
    'BU04a': 'mild', 'BU05a': 'mild', 'BU06a': 'mild', 'BU07a': 'moderate', 'BU08a': 'severe',
    'BU09a': 'moderate', 'BU10a': 'moderate', 'BU12a': 'mild', 'MBA01a': 'mild', 'MMA02a': 'mild',
    'MMA03a': 'mild', 'MMA05a': 'mild', 'MMA06a': 'severe', 'MMA08a': 'severe', 'MMA09a': 'mild',
    'MMA10a': 'very_severe', 'MMA11a': 'mild', 'MMA12a': 'mild', 'MMA13a': 'moderate', 'MMA14a': 'severe',
    'MMA15a': 'moderate', 'MMA16a': 'moderate', 'MMA17a': 'mild', 'MMA19a': 'mild', 'MMA20a': 'mild',
    'MMA22a': 'mild', 'MSU01a': 'mild', 'MSU01b': 'mild', 'MSU02a': 'mild', 'MSU02b': 'mild',
    'MSU03a': 'moderate', 'MSU03b': 'moderate', 'MSU04a': 'mild', 'MSU04b': 'mild', 'MSU05a': 'moderate',
    'MSU05b': 'moderate', 'MSU06a': 'mild', 'MSU06b': 'mild', 'MSU07a': 'moderate', 'MSU07b': 'moderate',
    'MSU08a': 'moderate', 'MSU08b': 'moderate', 'UCL01a': 'moderate', 'UCL02a': 'moderate',
    'UCL03a': 'severe', 'UCL04a': 'mild', 'UNH01a': 'mild', 'UNH01b': 'mild', 'UNH01c': 'mild',
    'UNH02a': 'very_severe', 'UNH02b': 'severe', 'UNH03a': 'mild', 'UNH03b': 'mild', 'UNH03c': 'mild',
    'UNH04a': 'severe', 'UNH04b': 'severe', 'UNH05a': 'mild', 'UNH05b': 'mild', 'UNH05c': 'mild',
    'UNH06a': 'mild', 'UNH07a': 'severe', 'UNH07b': 'moderate', 'UNH08a': 'moderate', 'UNH09a': 'moderate',
    'UNH09b': 'severe', 'UNH09c': 'severe', 'UNH10a': 'moderate', 'UNH11a': 'mild', 'adler01a': 'mild',
    'adler02a': 'moderate', 'adler03a': 'mild', 'adler04a': 'moderate', 'adler05a': 'moderate',
    'adler06a': 'severe', 'adler07a': 'mild', 'adler08a': 'mild', 'adler09a': 'mild',
    'adler10a': 'moderate', 'adler11a': 'very_severe', 'adler12a': 'mild', 'adler13a': 'moderate',
    'adler14a': 'mild', 'adler15a': 'moderate', 'adler16a': 'moderate', 'adler17a': 'mild',
    'adler18a': 'moderate', 'adler19a': 'very_severe', 'adler20a': 'mild', 'adler21a': 'mild',
    'adler22a': 'mild', 'adler23a': 'severe', 'adler24a': 'mild', 'adler25a': 'mild', 'cmu02b': 'moderate',
    'cmu03a': 'mild', 'elman01a': 'mild', 'elman01b': 'mild', 'elman02a': 'moderate',
    'elman03a': 'moderate', 'elman04a': 'mild', 'elman05a': 'mild', 'elman06a': 'severe',
    'elman07a': 'moderate', 'elman08a': 'severe', 'elman09a': 'moderate', 'elman10a': 'mild',
    'elman11a': 'moderate', 'elman11b': 'mild', 'elman12a': 'moderate', 'elman13a': 'mild',
    'elman14a': 'moderate', 'elman15a': 'mild', 'fridriksson01a': 'mild', 'fridriksson02a': 'mild',
    'fridriksson03a': 'severe', 'fridriksson03b': 'severe', 'fridriksson04a': 'mild',
    'fridriksson05a': 'mild', 'fridriksson06a': 'severe', 'fridriksson06b': 'moderate',
    'fridriksson07a': 'mild', 'fridriksson08b': 'very_severe', 'fridriksson09a': 'mild',
    'fridriksson09b': 'mild', 'fridriksson10a': 'moderate', 'fridriksson10b': 'moderate',
    'fridriksson11a': 'mild', 'fridriksson12a': 'moderate', 'fridriksson13a': 'moderate',
    'garrett01a': 'moderate', 'garrett02a': 'mild', 'kansas01a': 'very_severe', 'kansas02a': 'very_severe',
    'kansas03a': 'mild', 'kansas04a': 'mild', 'kansas05a': 'severe', 'kansas06a': 'very_severe',
    'kansas07a': 'mild', 'kansas08a': 'very_severe', 'kansas09a': 'moderate', 'kansas10a': 'moderate',
    'kansas11a': 'mild', 'kansas12a': 'severe', 'kansas13a': 'moderate', 'kansas14a': 'moderate',
    'kansas15a': 'mild', 'kansas16a': 'severe', 'kansas17a': 'moderate', 'kansas18a': 'mild',
    'kansas19a': 'mild', 'kansas20a': 'moderate', 'kansas21a': 'mild', 'kansas22a': 'moderate',
    'kansas23a': 'moderate', 'kempler02a': 'mild', 'kempler03a': 'moderate', 'kempler04a': 'moderate',
    'kurland01c': 'moderate', 'kurland01d': 'moderate', 'kurland02a': 'moderate', 'kurland02b': 'moderate',
    'kurland02c': 'mild', 'kurland02d': 'mild', 'kurland02e': 'mild', 'kurland03a': 'mild',
    'kurland03b': 'mild', 'kurland04a': 'mild', 'kurland04b': 'mild', 'kurland05a': 'mild',
    'kurland05b': 'mild', 'kurland06a': 'mild', 'kurland06b': 'mild', 'kurland07a': 'mild',
    'kurland07b': 'mild', 'kurland08a': 'mild', 'kurland08b': 'mild', 'kurland09a': 'mild',
    'kurland09b': 'mild', 'kurland10a': 'mild', 'kurland10b': 'mild', 'kurland12a': 'mild',
    'kurland12b': 'mild', 'kurland13a': 'moderate', 'kurland13b': 'moderate', 'kurland14a': 'mild',
    'kurland14b': 'mild', 'kurland15a': 'very_severe', 'kurland15b': 'very_severe',
    'kurland15c': 'very_severe', 'kurland16a': 'severe', 'kurland16b': 'severe', 'kurland17a': 'mild',
    'kurland17b': 'mild', 'kurland17c': 'mild', 'kurland18a': 'severe', 'kurland18b': 'severe',
    'kurland19a': 'moderate', 'kurland19b': 'moderate', 'kurland19c': 'moderate', 'kurland20a': 'moderate',
    'kurland21a': 'mild', 'kurland21b': 'mild', 'kurland21c': 'mild', 'kurland22a': 'severe',
    'kurland22b': 'severe', 'kurland23a': 'mild', 'kurland23b': 'mild', 'kurland23c': 'mild',
    'kurland24a': 'moderate', 'kurland24b': 'moderate', 'kurland24c': 'moderate', 'kurland25a': 'mild',
    'kurland25b': 'mild', 'kurland25c': 'mild', 'kurland26a': 'mild', 'kurland26b': 'mild',
    'kurland26c': 'mild', 'kurland27a': 'moderate', 'kurland27b': 'moderate', 'kurland27c': 'moderate',
    'kurland28a': 'mild', 'kurland28b': 'mild', 'kurland28c': 'mild', 'kurland29a': 'moderate',
    'kurland29b': 'moderate', 'kurland29c': 'moderate', 'scale01a': 'moderate', 'scale02a': 'moderate',
    'scale02b': 'mild', 'scale03a': 'severe', 'scale04a': 'moderate', 'scale05a': 'moderate',
    'scale05b': 'moderate', 'scale05c': 'mild', 'scale06a': 'mild', 'scale06b': 'mild', 'scale06c': 'mild',
    'scale06d': 'mild', 'scale07a': 'very_severe', 'scale08a': 'mild', 'scale09a': 'very_severe',
    'scale10a': 'moderate', 'scale11a': 'moderate', 'scale11b': 'moderate', 'scale12a': 'moderate',
    'scale12b': 'mild', 'scale12c': 'mild', 'scale12d': 'mild', 'scale13a': 'moderate',
    'scale14a': 'moderate', 'scale14c': 'moderate', 'scale15a': 'moderate', 'scale15b': 'moderate',
    'scale15c': 'moderate', 'scale15d': 'mild', 'scale16a': 'mild', 'scale17a': 'mild', 'scale17c': 'mild',
    'scale18a': 'moderate', 'scale18b': 'moderate', 'scale18c': 'moderate', 'scale18d': 'mild',
    'scale19a': 'moderate', 'scale20a': 'mild', 'scale21a': 'mild', 'scale22a': 'mild',
    'scale23a': 'moderate', 'scale24a': 'severe', 'scale25a': 'severe', 'scale26a': 'moderate',
    'scale27a': 'severe', 'scale28a': 'severe', 'scale30a': 'moderate', 'scale30b': 'moderate',
    'scale31a': 'moderate', 'scale32a': 'mild', 'scale33a': 'moderate', 'scale34a': 'mild',
    'scale35a': 'mild', 'scale36a': 'moderate', 'scale38a': 'moderate', 'star03a': 'mild', 'tap01a': 'mild',
    'tap02a': 'moderate', 'tap03a': 'severe', 'tap04a': 'mild', 'tap05a': 'mild', 'tap06a': 'severe',
    'tap07a': 'mild', 'tap08a': 'moderate', 'tap09a': 'very_severe', 'tap10a': 'moderate',
    'tap11a': 'moderate', 'tap12a': 'mild', 'tap13a': 'severe', 'tap14a': 'moderate', 'tap15a': 'moderate',
    'tap16a': 'moderate', 'tap17a': 'moderate', 'tap18a': 'mild', 'tap19a': 'moderate', 'tcu01a': 'mild',
    'tcu02b': 'severe', 'tcu03a': 'moderate', 'tcu05a': 'mild', 'tcu07a': 'moderate', 'tcu08a': 'moderate',
    'tcu09a': 'mild', 'tcu10b': 'mild', 'thompson01a': 'mild', 'thompson02a': 'mild',
    'thompson03a': 'moderate', 'thompson04a': 'moderate', 'thompson05a': 'moderate', 'thompson06a': 'mild',
    'thompson07a': 'mild', 'thompson07b': 'mild', 'thompson07c': 'mild', 'thompson08a': 'mild',
    'thompson09a': 'mild', 'thompson10a': 'mild', 'thompson11a': 'mild', 'thompson12a': 'mild',
    'thompson13a': 'mild', 'thompson14a': 'mild', 'tucson01a': 'mild', 'tucson02a': 'severe',
    'tucson03a': 'severe', 'tucson04a': 'mild', 'tucson06a': 'mild', 'tucson06b': 'mild',
    'tucson07a': 'moderate', 'tucson08a': 'moderate', 'tucson08b': 'mild', 'tucson09a': 'moderate',
    'tucson10a': 'mild', 'tucson11a': 'moderate', 'tucson12a': 'severe', 'tucson13a': 'moderate',
    'tucson14a': 'severe', 'tucson15a': 'severe', 'tucson15b': 'moderate', 'tucson16a': 'mild',
    'tucson18a': 'mild', 'tucson19a': 'severe', 'tucson20a': 'mild', 'tucson22a': 'moderate',
    'whiteside01a': 'mild', 'whiteside02a': 'moderate', 'whiteside03a': 'severe', 'whiteside04a': 'severe',
    'whiteside05a': 'mild', 'whiteside06a': 'mild', 'whiteside07a': 'mild', 'whiteside08a': 'moderate',
    'whiteside09a': 'mild', 'whiteside10a': 'moderate', 'whiteside11a': 'moderate',
    'whiteside12a': 'moderate', 'whiteside13a': 'mild', 'whiteside14a': 'moderate',
    'whiteside15a': 'moderate', 'whiteside16a': 'moderate', 'whiteside17a': 'mild', 'whiteside18a': 'mild',
    'whiteside19a': 'mild', 'whiteside20a': 'mild', 'williamson01a': 'moderate', 'williamson02a': 'mild',
    'williamson03a': 'moderate', 'williamson04a': 'moderate', 'williamson05a': 'mild',
    'williamson06a': 'moderate', 'williamson07a': 'mild', 'williamson08a': 'mild', 'williamson09a': 'mild',
    'williamson09b': 'mild', 'williamson10a': 'mild', 'williamson11a': 'moderate',
    'williamson12a': 'moderate', 'williamson12b': 'moderate', 'williamson13a': 'mild',
    'williamson14a': 'mild', 'williamson14b': 'mild', 'williamson15a': 'mild', 'williamson16a': 'moderate',
    'williamson17a': 'mild', 'williamson18a': 'mild', 'williamson19a': 'moderate',
    'williamson21a': 'very_severe', 'williamson23a': 'severe', 'williamson24a': 'moderate',
    'wozniak01a': 'mild', 'wozniak02a': 'mild', 'wozniak03a': 'moderate', 'wozniak04a': 'mild',
    'wozniak05a': 'mild', 'wozniak06a': 'mild', 'wozniak07a': 'moderate', 'wright201a': 'moderate',
    'wright202a': 'mild', 'wright203a': 'mild', 'wright204a': 'mild', 'wright205a': 'moderate',
    'wright206a': 'moderate', 'wright207a': 'moderate',
}

severity2spks = {}
for spk, severity in spk2severity.items():
    severity2spks.setdefault(severity, []).append(spk)


# ======================
#        Utility
# ======================


def get_utt(spk: str, start_time: int, end_time: int) -> str:
    return f"{spk}-{start_time}_{end_time}"


def utt2time(utt: str) -> (int, int):
    _, timestamps = utt.split("-")
    start, end = timestamps.split("_")
    return int(start), int(end)


def utt2spk(utt: str) -> str:
    return utt.split("-")[-2]
