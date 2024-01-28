import jiwer
import jiwer.transforms as tr
from packaging import version
import importlib.metadata as importlib_metadata


import unicodedata
import sys

ALL_PUNCTUATION = "".join((chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')))

SENTENCE_DELIMITER = ""
if version.parse(importlib_metadata.version("jiwer")) < version.parse("2.3.0"):

    class SentencesToListOfCharacters(tr.AbstractTransform):
      def __init__(self, sentence_delimiter: str = " "):
        self.sentence_delimiter = sentence_delimiter

      def process_string(self, s: str):
        return list(s)

      def process_list(self, inp):
        chars = []
        for sent_idx, sentence in enumerate(inp):
           chars.extend(self.process_string(sentence))
           if self.sentence_delimiter is not None and self.sentence_delimiter != "" and sent_idx < len(inp) - 1:
             chars.append(self.sentence_delimiter)
        return chars

    cer_transform = tr.Compose(
      [tr.RemoveMultipleSpaces(), tr.Strip(), SentencesToListOfCharacters(SENTENCE_DELIMITER)]
    )
else:
    cer_transform = tr.Compose(
      [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
        tr.ReduceToListOfListOfChars(),
      ]
    )


def compute_cer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    cer = jiwer.wer(reference, hypothesis, truth_transform=cer_transform, hypothesis_transform=cer_transform)
    return cer


def compute_wer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    wer = jiwer.wer(reference, hypothesis) 
    return wer

def normalize_text(text):
    # remove ponctuation
    text = text.translate(str.maketrans('', '', ALL_PUNCTUATION))
    text = text.lower()
    text = ' '.join(text.split())
    if not text:
      text = " "
    return text

