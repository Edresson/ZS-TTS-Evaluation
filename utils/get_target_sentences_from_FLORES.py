import os
from collections import OrderedDict
import random
import re
random.seed(26)

tot_num_of_sentences = 240 # (4 * 58) - 3 samples for each built-in speaker
min_len = 1
max_len = 58


dataset_path = "/raid/datasets/FLORES/floresp-v2.0-alpha.1/devtest/"

language_map = {
    "pt": "devtest.por_Latn",
    "es": "devtest.spa_Latn",
    "fr": "devtest.fra_Latn",
    "de": "devtest.deu_Latn",
    "it": "devtest.ita_Latn",
    "pl": "devtest.pol_Latn",
    "tr": "devtest.tur_Latn",
    "ru": "devtest.rus_Cyrl",
    "nl": "devtest.nld_Latn",
    "cs": "devtest.ces_Latn",
    "ar": "devtest.arb_Arab",
    "zh": "devtest.cmn_Hans",
    "hu": "devtest.hun_Latn",
    "ko": "devtest.kor_Hang",
    "ja": "devtest.jpn_Jpan",
    "fi": "devtest.fin_Latn",
    "hi": "devtest.hin_Deva",
    "sv": "devtest.swe_Latn",
}

lines = open(os.path.join(dataset_path, "devtest.eng_Latn")).readlines()

print("Number of words:", len(lines))
lines_dict = []
for i in range(len(lines)):
    text = lines[i].replace("\n", "").replace('"', '')
    # print(text)
    lines_dict.append({"text": text, "idx": i, "len": len(text.split(" "))})



# filter text with numbers to avoid issues with previous models that do not handle numbers
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

lines_dict_new = []
for d in lines_dict:
    if not has_numbers(d["text"]):
        text_without_abbreviations = re.sub(r"\b[A-Z]{2,}\b", "", d["text"])
        # filter text with abbreviations because some previous models cant deal with it
        if len(text_without_abbreviations) == len(d["text"]) and "(" not in d["text"] and "[" not in d["text"]:
            lines_dict_new.append(d)
lines_dict = lines_dict_new


# sort by len
lines_dict = sorted(lines_dict, key=lambda d: d['len']) 

tot_len = len(lines_dict)
slice = len(lines_dict)//3
num_samples_each = tot_num_of_sentences//3

indexes = []

s_dict = [d["idx"] for d in lines_dict[:slice]]
m_dict = [d["idx"] for d in lines_dict[slice:slice+slice]]
l_dict = [d["idx"] for d in lines_dict[-slice:]]

random.seed(26)
indexes += random.sample(list(s_dict), num_samples_each)
indexes += random.sample(list(m_dict), num_samples_each)
indexes += random.sample(list(l_dict), num_samples_each)

def get_sentences_from_idx(lines_dict, indexes):
    sentences = []
    for idx in indexes:
        for d in lines_dict:
            if d["idx"] == idx:
                sentences.append(d)
                # print(d["text"])

    sentences = sorted(sentences, key=lambda d: d['len'])

    sentences_text = [d["text"] for d in sentences]
    return sentences, sentences_text

sentences_info, sentences_en = get_sentences_from_idx(lines_dict, indexes)
print(f"en_sentences={sentences_en}")

for lang in language_map:
    sentences = []
    lang_lines = open(os.path.join(dataset_path, language_map[lang])).readlines()
    for d in sentences_info:
        text = lang_lines[d["idx"]].replace("\n", "").replace('"', '')
        # english text do not have [] () so if it exist in translation it means that it is the english word -- it happens sometimes in the corpora

        text = re.sub("[\(\[].*?[\)\]]", "", text)
        sentences.append(text)
        # print(d["text"], text)
    print(f"{lang}_sentences={sentences}")