import os
import re
import textwrap
import torch
import librosa

from num2words import num2words
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from transformers import pipeline
# from TTS.tts.layers.xtts.tokenizer import expand_numbers_multilingual
from faster_whisper import WhisperModel


from functools import cached_property

import pypinyin
import torch
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from num2words import num2words
from spacy.lang.ar import Arabic
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.ja import Japanese
from spacy.lang.zh import Chinese
from tokenizers import Tokenizer

# from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words
from utils.zh_num2words import TextNorm as zh_num2words

def get_spacy_lang(lang):
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "es":
        return Spanish()
    else:
        # For most languages, Enlish does the job
        return English()


def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = get_spacy_lang(lang)
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                # if the last sentence + the current sentence is less than the text_split_length
                # then add the current sentence to the last sentence
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
                # if the current sentence is greater than the text_split_length
                for line in textwrap.wrap(
                    str(sentence),
                    width=text_split_length,
                    drop_whitespace=True,
                    break_on_hyphens=False,
                    tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(str(sentence))

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # ("sig.ra", "signora"),
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "zh": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("ing", "inženýr"),  # engineer
            ("p", "pan"),  # Could also map to pani for woman but no easy way to do it
            # Other abbreviations would be specialized and not as common.
        ]
    ],
    "ru": [
        (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),  # Mrs.
            ("г-н", "господин"),  # Mr.
            ("д-р", "доктор"),  # doctor
            # Other abbreviations are less common or specialized.
        ]
    ],
    "nl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),  # Mr.
            ("mevr", "mevrouw"),  # Mrs.
            ("dr", "dokter"),  # doctor
            ("jhr", "jonkheer"),  # young lord or nobleman
            # Dutch uses more abbreviations, but these are the most common ones.
        ]
    ],
    "tr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"),  # Mr.
            ("byk", "büyük"),  # büyük
            ("dr", "doktor"),  # doctor
            # Add other Turkish abbreviations here if needed.
        ]
    ],
    "hu": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("b", "bácsi"),  # Mr.
            ("nőv", "nővér"),  # nurse
            # Add other Hungarian abbreviations here if needed.
        ]
    ],
    "ko": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Korean doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
    "es": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " y "),
            ("@", " arroba "),
            ("%", " por ciento "),
            ("#", " numeral "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " et "),
            ("@", " arobase "),
            ("%", " pour cent "),
            ("#", " dièse "),
            ("$", " dollar "),
            ("£", " livre "),
            ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " und "),
            ("@", " at "),
            ("%", " prozent "),
            ("#", " raute "),
            ("$", " dollar "),
            ("£", " pfund "),
            ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " arroba "),
            ("%", " por cento "),
            ("#", " cardinal "),
            ("$", " dólar "),
            ("£", " libra "),
            ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " chiocciola "),
            ("%", " per cento "),
            ("#", " cancelletto "),
            ("$", " dollaro "),
            ("£", " sterlina "),
            ("°", " gradi "),
        ]
    ],
    "pl": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " i "),
            ("@", " małpa "),
            ("%", " procent "),
            ("#", " krzyżyk "),
            ("$", " dolar "),
            ("£", " funt "),
            ("°", " stopnie "),
        ]
    ],
    "ar": [
        # Arabic
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " و "),
            ("@", " على "),
            ("%", " في المئة "),
            ("#", " رقم "),
            ("$", " دولار "),
            ("£", " جنيه "),
            ("°", " درجة "),
        ]
    ],
    "zh": [
        # Chinese
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "),
            ("@", " 在 "),
            ("%", " 百分之 "),
            ("#", " 号 "),
            ("$", " 美元 "),
            ("£", " 英镑 "),
            ("°", " 度 "),
        ]
    ],
    "cs": [
        # Czech
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " a "),
            ("@", " na "),
            ("%", " procento "),
            ("#", " křížek "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " stupně "),
        ]
    ],
    "ru": [
        # Russian
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " и "),
            ("@", " собака "),
            ("%", " процентов "),
            ("#", " номер "),
            ("$", " доллар "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "nl": [
        # Dutch
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " en "),
            ("@", " bij "),
            ("%", " procent "),
            ("#", " hekje "),
            ("$", " dollar "),
            ("£", " pond "),
            ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "),
            ("@", " at "),
            ("%", " yüzde "),
            ("#", " diyez "),
            ("$", " dolar "),
            ("£", " sterlin "),
            ("°", " derece "),
        ]
    ],
    "hu": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " és "),
            ("@", " kukac "),
            ("%", " százalék "),
            ("#", " kettőskereszt "),
            ("$", " dollár "),
            ("£", " font "),
            ("°", " fok "),
        ]
    ],
    "ko": [
        # Korean
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 그리고 "),
            ("@", " 에 "),
            ("%", " 퍼센트 "),
            ("#", " 번호 "),
            ("$", " 달러 "),
            ("£", " 파운드 "),
            ("°", " 도 "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),  # In Czech, a dot is often used after the number to indicate ordinals.
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang=lang if lang != "cs" else "cz")


def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = num2words(amount, to="currency", currency=currency, lang=lang if lang != "cs" else "cz")

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents[lang])
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return num2words(int(m.group(1)), ordinal=True, lang=lang if lang != "cs" else "cz")


def _expand_number(m, lang="en"):
    return num2words(int(m.group(0)), lang=lang if lang != "cs" else "cz")


def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh":
        text = zh_num2words()(text)
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
        except:
            pass
        if lang != "tr":
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text

def custom_expand_numbers_multilingual(text, lang):
    # if coqui TTS number expands fails, uses num2words
    try:
        text = expand_numbers_multilingual(text, lang)
    except:
        if lang == "cs":
            lang = "cz"

        numbers = re.findall(r'\d+', text)
        # Transliterate the numbers to text
        for num in numbers:
            try:
                transliterated_num = ''.join(num2words(int(num), lang=lang))
            except:
                transliterated_num = num
            text = text.replace(num, transliterated_num, 1)
    return text


class FasterWhisperSTT(object):
    def __init__(self, model_name="large-v3", use_cuda=False) -> None:
        self.model = WhisperModel(model_name, device='cuda' if use_cuda else 'cpu', compute_type="float16")
        self.segments = None

    def transcribe_audio(self, audio, language=None):
        segments, _ = self.model.transcribe(audio, beam_size=5, language=language)
        segments = list(segments)
        self.segments = segments
        transcription = "".join([segment.text for segment in segments])
        # convert number to words
        transcription = custom_expand_numbers_multilingual(transcription, lang=language)
        return transcription

    def get_segments(self):
        return self.segments


# ToDo: uses transformers pipeline
class Wav2VecSTT(object):
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self", use_cuda=False, sr=16000) -> None:
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.sr = sr
        if use_cuda:
            self.model = self.model.cuda()

    def cuda(self):
        self.model = self.model.cuda()

    def cpu(self):
        self.model = self.model.cpu()

    def transcribe_audio(self, audio, language=None):
        input_audio, _ = librosa.load(audio, sr=self.sr)
        input_values = self.tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
            # arg softmax  t oget the most probablies tokens
            predicted_ids = torch.argmax(logits, dim=-1)

        # decode
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        # convert number to words
        transcription = custom_expand_numbers_multilingual(transcription,  lang=language)
        return transcription

    def get_segments(self):
        return None


class MMSWav2VecSTT(object):
    def __init__(self, model_name="facebook/mms-1b-all", use_cuda=False) -> None:
        self.pipe = pipeline(model=model_name, device='cuda:0' if use_cuda else 'cpu')

    def transcribe_audio(self, audio, language=None):
        out = self.pipe(audio)
        transcription = out["text"]
        # convert number to words
        transcription = custom_expand_numbers_multilingual(transcription, lang=language)
        return transcription

    def get_segments(self):
        return None
