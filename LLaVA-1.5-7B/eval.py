import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from utils import *


model_path = "..."
data_dir = "/datasets/..."


processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16
)


samples = load_test(data_dir)


correct = 0
total = 0


for s in tqdm(samples, desc="Evaluating Test"):
    prompt = processor.apply_chat_template(s["prompt"], add_generation_prompt=True)
    inputs = processor(images=s["images"], text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    generated = processor.decode(output_ids[0], skip_special_tokens=True)
    pred = parse_answer(generated)
    ans = s["answer"]
    total += 1

    if pred == ans:
        correct += 1


accuracy = correct / total if total > 0 else 0.0
print(f"Correct: {correct}")
print(f"Total evaluated: {total}")
print(f"Accuracy: {accuracy:.4f}")
