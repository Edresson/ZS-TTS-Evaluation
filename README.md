# ZS-TTS-Evaluation

This repository gathers our efforts to evaluate/compare the current Zero-shot Multi-speaker TTS (ZS-TTS) systems using objective metrics.

To compare the models we have used 240 sentences for each supported language from [FLORES+](https://github.com/openlanguagedata/flores).
 The sentences were chosen randomly from the $devtest$ subset. We have chosen the FLORES+ dataset because it has parallel translations for all languages supported by most of the ZS-TTS models. 
 In this way, we can compare all the language results using the same vocabulary. To test the ZS-TTS capability we decided to use all 20 speakers (10M and 10F) from the clean subset of the [DAPS dataset](https://zenodo.org/records/4660670). 
 For each speaker, we randomly selected one audio segment between 3 and 8 seconds to use as a reference during the test sentence generation. 
 We have used these samples to evaluate all languages, that way for non-English languages the models are compared in a cross-lingual way.


## Metrics

### UTMOS
Following [HierSpeech++ paper](https://arxiv.org/abs/2311.12454), we have used the [UTMOS model](https://arxiv.org/abs/2204.02152) to predict the Naturalness Mean Opinion Score (nMOS). 
In the HierSpeech++ paper, the authors have used the open-source version of UTMOS\footnote{https://github.com/tarepan/SpeechMOS}, and the presented results of human nMOS and UTMOS are almost aligned. 
Although this can not be considered an absolute evaluation metric, it can be used to easily compare models in quality terms. 

### SECS
To compare the similarity between the synthesized voice and the original speaker, we compute the [Speaker Encoder Cosine Similarity (SECS)](https://arxiv.org/abs/2104.05557).
We have computed SECS using the [ECAPA2](https://huggingface.co/Jenthe/ECAPA2) speaker encoder, because it achieved recently speaker verification SOTA in VoxCeleb1 test sets.

### CER
Following previous works, we evaluate pronunciation accuracy using an ASR model. For it, we have used the [Whisper Large v3 model](https://huggingface.co/openai/whisper-large-v3). 
Additionally, considering that our model was trained in languages that do not use spaces to separate words (e.g. Chinese), we decided to use the Character Error Rate (CER) instead of the Word Error Rate (WER) because in this way all languages are evaluated with the same metric. We also removed all text punctuation before computing CER.


## Reference
To reference this repository in your work please use the following reference:
```
@inproceedings{casanova24_interspeech,
  title     = {XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model},
  author    = {Edresson Casanova and Kelly Davis and Eren Gölge and Görkem Göknar and Iulian Gulea and Logan Hart and Aya Aljafari and Joshua Meyer and Reuben Morais and Samuel Olayemi and Julian Weber},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4978--4982},
  doi       = {10.21437/Interspeech.2024-2016},
  issn      = {2958-1796},
}
```
