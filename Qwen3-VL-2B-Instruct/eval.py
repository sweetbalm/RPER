import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


model_path = "..."
data_dir = "datasets/..."


model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto"
)


processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=False
)


from utils import *


samples = load_test(data_dir)

correct = 0
total = 0


for s in tqdm(samples, desc="Evaluating Test"):
    inputs = processor.apply_chat_template(
        s["messages"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    pred = parse_answer(output_text[0])
    ans = s["answer"]
    total += 1

    if pred == ans:
        correct += 1


accuracy = correct / total if total > 0 else 0.0
print(f"Correct: {correct}")
print(f"Total evaluated: {total}")
print(f"Accuracy: {accuracy:.4f}")
