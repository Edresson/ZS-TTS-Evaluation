

# import sys
# sys.path.append('StyleTTS2')

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import pandas as pd
import os

import random
from glob import glob
from tqdm import tqdm

random.seed(0)

EVAL_PATH = "/raid/edresson/dev/Paper/TTS-evaluation-public/Evaluation/"



samples_files = glob(f'{EVAL_PATH}/**/custom_generated_sentences.csv', recursive=True)

for SAMPLES_CSV in tqdm(samples_files):
    metric_file = os.path.join(os.path.dirname(SAMPLES_CSV), "metrics.csv")
    # if os.path.isfile(metric_file):
    #     continue
    os.system(f"python eval_TTS.py --csv_path {SAMPLES_CSV}")
    print("Done and saved at: ", metric_file)

