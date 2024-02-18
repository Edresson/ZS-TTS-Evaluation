import random
import uuid
import gradio as gr
import pandas as pd
from datetime import datetime

lang = "en"
model_a_name = "XTTS_v2.0.2"
model_b_name = "Mega-TTS2"
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
filename = f"./SCMOS_{model_a_name}_{model_b_name}_{get_timestamp()}.csv"

with open(filename, "w") as file:
    file.write("timestamp,uuid,score\n")

iteration = 0

def sample_random(state):
    global iteration
    row = merged.iloc[iteration%len(merged)]
    state["inverted"] = True if random.randint(0,1) == 1 else False
    A = row[model_a_name] if not state["inverted"] else row[model_b_name]
    B = row[model_a_name] if state["inverted"] else row[model_b_name]
    ref = row["speaker_reference"]
    print("Iteration: ", iteration)
    iteration += 1
    return A, B, ref
    
def vote(score, state_vars, a_audio, b_audio):
    if a_audio is None or b_audio is None:
        return (None, None)
    with open(filename, "a+") as f:
        f.write(f"{get_timestamp()},{state_vars['uid']},{str(score) if not state_vars['inverted'] else str(-score)}\n")
    return sample_random(state_vars)

def init_state():
    return {"inverted": False, "uid": str(uuid.uuid4())}
    
with gr.Blocks() as demo:
    state_vars = gr.State(init_state)
    with gr.Column() as col1:
        gr.Markdown("## SCMOS: Similarity Comparative Model Opinion Score\n\n"
                    "This tool is designed to collect human opinions on the voice cloning ability of two text-to-speech models.\n\n"
                    "You will be presented with two audio clips, A and B, and you will be asked to score how close A sounds to the reference voice compared to B.\n\n"
                    "When you submit a vote, the next pair of audio clips will be presented to you acompanied by the matching reference.\n\n"
                    "**Please use headphones if possible and rate at least 8 pairs of audio clips.**\n\n"
                    "When you are ready, click the start button below to start the evaluation.")
        start_btn = gr.Button(value="Start")
    with gr.Column() as col2:
        gt_audio = gr.Audio(label="Ground truth", autoplay=True)
        a_audio = gr.Audio(label="A", interactive=False)
        b_audio = gr.Audio(label="B", interactive=False)
        gr.Markdown("## Score how close A sounds to the audio reference compared to B.\n\n"
                    "**Only judge similarity (timber, accent etc...), not audio quality or naturalness.**\n\n"
                    "- 2 <=> A is closer to reference than B\n"
                    "- 1 <=> A is slightly closer to reference than B\n"
                    "- 0 <=> Both are equaly close to reference than B\n"
                    "- -1 <=> A is slightly farther to reference than B\n"
                    "- -2 <=> A is farther to reference than B\n\n\n"
                    '<div style="display: flex; justify-content: space-between";margin-top: 20px;><div style="text-align: left">B is more similar</div><div style="text-align: right">A is more similar</div></div>')
        score = gr.Slider(minimum=-2, maximum=2, step=1, value=0, info="Score")
        vote_btn = gr.Button(value="Submit vote")

    start_btn.click(
        fn=sample_random,
        inputs=[state_vars],
        outputs=[a_audio, b_audio, gt_audio],
    )

    vote_btn.click(
        fn=vote,
        inputs=[score, state_vars, a_audio, b_audio],
        outputs=[a_audio, b_audio],
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        server_port=3008,
        server_name="0.0.0.0",
    )