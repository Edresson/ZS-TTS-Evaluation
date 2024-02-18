import random
import gradio as gr
import pandas as pd
from datetime import datetime

mode = "CMOS"
lang = "en"
model_a_name = "Mega-TTS2"
model_b_name = "XTTS_v2.0.2"
# HierSpeech++

model_a = pd.read_csv(f"Evaluation/{model_a_name}/custom_generated_sentences.csv")
model_a = model_a[model_a["language"]==lang]
model_a = model_a.rename(columns={'generated_wav': model_a_name})
model_a = model_a.drop(columns=["language"])

model_b = pd.read_csv(f"Evaluation/{model_b_name}/custom_generated_sentences.csv")
model_b = model_b[model_b["language"]==lang]
model_b = model_b.rename(columns={'generated_wav': model_b_name})
model_b = model_b.drop(columns=["language"])

merged = pd.merge(model_a, model_b, on=["speaker_reference", "text"])
merged = merged.sample(frac=1).reset_index(drop=True)
assert len(merged) == len(model_a) == len(model_b) == 240


get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = f"./{mode}_{model_a_name}_{model_b_name}_{get_timestamp()}.csv"

with open(filename, "w") as file:
    file.write("timestamp,score\n")

iteration = 0

def sample_random(state):
    global iteration
    row = merged.iloc[iteration%len(merged)]
    state["inverted"] = True if random.randint(0,1) == 1 else False
    A = row[model_a_name] if not state["inverted"] else row[model_b_name]
    B = row[model_a_name] if state["inverted"] else row[model_b_name]
    print("Iteration: ", iteration)
    iteration += 1
    return A, B
    
def vote(score, state_vars):
    with open(filename, "a+") as f:
        f.write(f"{get_timestamp()},{str(score) if not state_vars['inverted'] else str(-score)}\n")
    return sample_random(state_vars)
    
with gr.Blocks() as demo:
    state_vars = gr.State({"inverted": False})
    with gr.Column() as col1:
        gr.Markdown("### CMOS: Comparative Model Opinion Score\n\n"
                    "This tool is designed to collect human opinions on the quality of two TTS models.\n\n"
                    "You will be presented with two audio clips, A and B, and you will be asked to score how A sounds compared to B.\n\n"
                    "When you submit a vote, the next pair of audio clips will be presented to you.\n\n"
                    "When you are ready, click the start button below to start the evaluation.")
        start_btn = gr.Button(value="Start")
    with gr.Column() as col2:
        #gt_audio = gr.Audio(label="Ground truth", autoplay=True)
        a_audio = gr.Audio(label="A")
        b_audio = gr.Audio(label="B")
        gr.Markdown("### Score how A sounds compared to B:\n\n"
                    "- 2 <=> A is better\n"
                    "- 1 <=> A is slightly better\n"
                    "- 0 <=> Both are equal\n"
                    "- -1 <=> A is slightly worse\n"
                    "- -2 <=> A is worse")
        score = gr.Slider(minimum=-2, maximum=2, step=1, value=0, info="Score")
        vote_btn = gr.Button(value="Submit vote")

    start_btn.click(
        fn=sample_random,
        inputs=[state_vars],
        outputs=[a_audio, b_audio],
    )

    vote_btn.click(
        fn=vote,
        inputs=[score, state_vars],
        outputs=[a_audio, b_audio],
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        server_port=3008,
        server_name="0.0.0.0",
    )