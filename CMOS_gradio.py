import random
import uuid
import gradio as gr
import pandas as pd
from datetime import datetime

lang = "en"
model_a_name = "XTTS_v2.0.2"
model_b_name = "Mega-TTS2"
model_c_name = "HierSpeech++"
# HierSpeech++

model_a = pd.read_csv(f"Evaluation/{model_a_name}/custom_generated_sentences.csv")
model_a = model_a[model_a["language"]==lang]
model_a = model_a.rename(columns={'generated_wav': model_a_name})
model_a = model_a.drop(columns=["language"])

model_b = pd.read_csv(f"Evaluation/{model_b_name}/custom_generated_sentences.csv")
model_b = model_b[model_b["language"]==lang]
model_b = model_b.rename(columns={'generated_wav': model_b_name})
model_b = model_b.drop(columns=["language"])

model_c = pd.read_csv(f"Evaluation/{model_c_name}/custom_generated_sentences.csv")
model_c = model_c[model_c["language"]==lang]
model_c = model_c.rename(columns={'generated_wav': model_c_name})
model_c = model_c.drop(columns=["language"])

merged = pd.merge(model_a, model_b, on=["speaker_reference", "text"])
merged = pd.merge(merged, model_c, on=["speaker_reference", "text"])
merged = merged.sample(frac=1).reset_index(drop=True)
assert len(merged) == len(model_a) == len(model_b) == len(model_c) == 240


get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = f"./CMOS_results_{get_timestamp()}.csv"

with open(filename, "w") as file:
    file.write("timestamp,uuid,score,against\n")

iteration = 0

def sample_random(state):
    global iteration
    row = merged.iloc[iteration%len(merged)]
    state["current_comparison"] = model_b_name if state["current_comparison"] == model_c_name else model_c_name
    state["inverted"] = True if random.randint(0,1) == 1 else False
    
    Left = row[model_a_name] if not state["inverted"] else row[state["current_comparison"]]
    Right = row[model_a_name] if state["inverted"] else row[state["current_comparison"]]
    print("Left: ", Left)
    print("Right: ", Right)
    print("Inverted: ", state["inverted"])
    print("Current comparison: ", state["current_comparison"])
    print("Iteration: ", iteration)
    iteration += 1
    return Left, Right
    
def vote(score, state_vars, left_audio, right_audio):
    if left_audio is None or right_audio is None:
        return (None, None)
    with open(filename, "a+") as f:
        f.write(f"{get_timestamp()},{state_vars['uid']},{str(score) if state_vars['inverted'] else str(-score)},{state_vars['current_comparison']}\n")
    return sample_random(state_vars)

def init_state():
    return {
        "inverted": False,
        "current_comparison": model_b_name if random.randint(0,1) == 1 else model_c_name,
        "uid": str(uuid.uuid4())}
    
with gr.Blocks() as demo:
    state_vars = gr.State(init_state)
    with gr.Column() as col1:
        gr.Markdown("## CMOS: Comparative Model Opinion Score\n\n"
                    "This tool is designed to collect human opinions on the naturalness of two text-to-speech models.\n\n"
                    "You will be presented with two audio clips, **Left** and **Right**, and you will be asked to score how sounds compared to each other.\n\n"
                    "The score should reflect your preference **in terms of prosody and naturalness.** (don't focus on audio quality)\n\n"
                    "When you submit a vote, the next pair of audio clips will be presented to you.\n\n"
                    "**Please use headphones if possible and rate at least 8 pairs of audio clips.**\n\n"
                    "When you are ready, click the start button below to start the evaluation.")
        start_btn = gr.Button(value="Start")
    with gr.Column() as col2:
        gr.Markdown("## Score how Right sounds compared to Left in terms of prosody and naturalness:\n\n"
                    "- 2  <=> Right is better\n"
                    "- 1  <=> Right is slightly better\n"
                    "- 0  <=> Both are equal\n"
                    "- -1 <=> Left is slightly better\n"
                    "- -2 <=> Left is better\n\n\n"
                    '<div style="display: flex; justify-content: center;margin-top: 20px;align-items: center;"><div style="text-align: left;font-size: medium;">Left is better</div><svg fill="#ffffff" height="5vh" width="200px" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 512.04 512.04" xml:space="preserve" stroke="#ffffff"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g> <g> <path d="M508.933,248.353L402.267,141.687c-4.267-4.053-10.987-3.947-15.04,0.213c-3.947,4.16-3.947,10.667,0,14.827 l88.427,88.427H36.4l88.427-88.427c4.053-4.267,3.947-10.987-0.213-15.04c-4.16-3.947-10.667-3.947-14.827,0L3.12,248.353 c-4.16,4.16-4.16,10.88,0,15.04L109.787,370.06c4.267,4.053,10.987,3.947,15.04-0.213c3.947-4.16,3.947-10.667,0-14.827 L36.4,266.593h439.147L387.12,355.02c-4.267,4.053-4.373,10.88-0.213,15.04c4.053,4.267,10.88,4.373,15.04,0.213 c0.107-0.107,0.213-0.213,0.213-0.213l106.667-106.667C513.093,259.34,513.093,252.513,508.933,248.353z"></path> </g> </g> </g></svg><div style="text-align: right;font-size: medium;">Right is better</div></div>')
        with gr.Row() as row1:
            left_audio = gr.Audio(label="Left", interactive=False)
            score = gr.Slider(minimum=-2, maximum=2, step=1, value=0, info="Score")
            right_audio = gr.Audio(label="Right", interactive=False)
        vote_btn = gr.Button(value="Submit vote")

    start_btn.click(
        fn=sample_random,
        inputs=[state_vars],
        outputs=[left_audio, right_audio],
    )

    vote_btn.click(
        fn=vote,
        inputs=[score, state_vars, left_audio, right_audio],
        outputs=[left_audio, right_audio],
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        server_port=3008,
        server_name="0.0.0.0",
    )