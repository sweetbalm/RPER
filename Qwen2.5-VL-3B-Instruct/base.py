import os
import json
import re
import random
import torch
from PIL import Image
from datasets import Dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from trl import GRPOConfig, GRPOTrainer


model_path = "..."
output_dir = "..."
data_dir = "/datasets/..."

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=False
)

from utils import *


data = load_train(data_dir)


training_args = GRPOConfig(
    output_dir=output_dir,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=7e-6,
    max_prompt_length=None,
    max_completion_length=512,
    temperature=1.0,
    logging_steps=50,
    bf16=True,
    tf32=True,
    loss_type="grpo"
)


trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    processing_class=processor,
    reward_funcs=[sqa_reward_func, reason_reward_func]
)

print("Starting GRPO training...")
trainer.train()
trainer.save_model(output_dir)
print(f"Training completed.")
