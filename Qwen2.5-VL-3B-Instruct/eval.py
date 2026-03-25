import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


model_path = "..."
data_dir = "/datasets/..."


processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=False
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto"
)

from utils import *


samples = load_test(data_dir)

correct = 0
total = 0


for s in tqdm(samples, desc="Evaluating Test"):
    image_inputs, _ = process_vision_info(s["messages"])
    text = processor.apply_chat_template(
        s["messages"], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=_,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
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
