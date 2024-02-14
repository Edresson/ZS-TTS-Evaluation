import os
import torch
import random
import numpy as np
import transformers

from utils.stt import FasterWhisperSTT
import torchaudio
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from utils.generic_utils import compute_cer, normalize_text, torch_rms_norm

from utils.export import export_metrics

import pandas as pd
from tqdm import tqdm
import argparse
from argparse import RawTextHelpFormatter
import librosa
import tempfile
from pydub import AudioSegment

# set seed to ensures reproducibility
def set_seed(random_seed=1234):
    # set deterministic inference
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    transformers.set_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)

set_seed()

CUDA_AVAILABLE = torch.cuda.is_available()
device = "cuda" if CUDA_AVAILABLE else "cpu"

## SECS utils
# automatically checks for cached file, optionally set `cache_dir` location
ecapa2_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(ecapa2_file, map_location='cpu').to(device)

def get_ecapa2_spk_embedding(path, ref_dBFS=None, model_sr=16000):
    audio, sr = torchaudio.load(path)
    # sample rate of 16 kHz expected
    if sr != model_sr:
        audio = torchaudio.functional.resample(audio, sr, model_sr)

    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    if ref_dBFS is not None:
        audio = torch_rms_norm(audio, db_level=ref_dBFS)

    # compute speaker embedding
    embed = ecapa2(audio.to(device))
    # ensures that l2 norm is applied on output
    embed = torch.nn.functional.normalize(embed, p=2, dim=1)
    return embed.cpu().detach().squeeze().numpy()


## UTMOS utils

# uses UTMOS (https://arxiv.org/abs/2204.02152) Open source (https://github.com/tarepan/SpeechMOS) following https://arxiv.org/abs/2311.12454
mos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)

def compute_UTMOS(path, ref_dBFS):
    # audio, sr = torchaudio.load(path)
    audio, sr = librosa.load(path, sr=None, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0)
    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    audio = torch_rms_norm(audio, db_level=ref_dBFS)
    # predict UTMOS
    score = mos_predictor(audio.to(device), sr).item()
    return score


def compute_metrics(tts_wav, ref_wav, gt_text, ref_speaker_embedding=None,  language="en", debug=False, ref_dBFS=None):
    language = language.split("-")[0]  # remove the region
    transcription = transcriber.transcribe_audio(tts_wav, language=language)

    # normalize texts - removing ponctuations
    gt_text_normalized = normalize_text(gt_text)
    transcription_normalized = normalize_text(transcription)

    # compute WER
    cer_tts = compute_cer(gt_text_normalized, transcription_normalized) * 100

    # compute UTMOS
    mos = compute_UTMOS(tts_wav, ref_dBFS)

    # compute SECS using ECAPA2 model
    gen_speaker_embedding = get_ecapa2_spk_embedding(tts_wav, ref_dBFS)
    gt_speaker_embedding = torch.FloatTensor(ref_speaker_embedding).unsqueeze(0)
    gen_speaker_embedding = torch.FloatTensor(gen_speaker_embedding).unsqueeze(0)
    secs = torch.nn.functional.cosine_similarity(gt_speaker_embedding, gen_speaker_embedding).item()

    if debug:
        print("Speaker Reference Path", ref_wav)
        print("TTS Audio Path:", tts_wav)
        print("Language:", language)
        print("GT text:", gt_text_normalized)
        print("Transcription:", transcription_normalized)
        print("CER:", cer_tts)
        print("UTMOS MOS:", mos)
        print("SECS:", secs)

    meta_dict = {"CER": cer_tts, "UTMOS": mos, "SECS": secs, "Language": language}
    meta_dict_full = meta_dict.copy()
    meta_dict_full["Speaker Reference Path"] = ref_wav
    meta_dict_full["Audio Path"] = tts_wav
    meta_dict_full["Num. Chars"] = len(gt_text)
    meta_dict_full["Ground Truth Text"] = gt_text
    meta_dict_full["Transcription"] = transcription
    meta_dict_full["Ground Truth Normalized Text"] = gt_text_normalized
    meta_dict_full["Normalized Transcription"] = transcription_normalized
    return meta_dict, meta_dict_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" ",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Input csv file path.",
        required=True,
    )


    args = parser.parse_args()
    
    # load test samples
    DF = pd.read_csv(args.csv_path)

    out_csv_path=os.path.join(os.path.dirname(args.csv_path), "metrics.csv")

    # uses whisper large-v3 to have accurated transcriptions in multiples languages
    transcriber = FasterWhisperSTT("large-v3", use_cuda=CUDA_AVAILABLE)

    metadata_list = []
    metadata_list_full = []
    missing_files = 0
    # group by using speaker reference
    df_speaker = DF.groupby('speaker_reference')
    for ref_wav, df_group in tqdm(df_speaker):
        speaker_embedding = get_ecapa2_spk_embedding(ref_wav)
        # get reference dBFS
        ref_dBFS = AudioSegment.from_file(ref_wav).dBFS

        for _, row in df_group.iterrows():
            tts_wav_path = row["generated_wav"]
            if not os.path.isfile(tts_wav_path):
                print(f"WARNING: The file {tts_wav_path} doesn't exits !")
                missing_files += 1
                if missing_files > 3:
                    raise RuntimeError(f"More than 3 wave files are missing for the CSV {args.csv_path}!! Please check it, because It can compromise the evaluation.")
                continue

            meta_dict, meta_dict_full = compute_metrics(tts_wav_path, ref_wav, row["text"], ref_speaker_embedding=speaker_embedding, language=row["language"], ref_dBFS=ref_dBFS)
            metadata_list.append(meta_dict)
            metadata_list_full.append(meta_dict_full)

    # save final metrics
    export_metrics(metadata_list, metadata_list_full, out_csv_path)
