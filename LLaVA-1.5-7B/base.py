import os
import json
import re
import torch
from PIL import Image
from datasets import Dataset
from transformers import LlavaForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer


model_path = "..."
output_dir = "..."
data_dir = "..."

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)


from utils import *


data = load_train(data_dir)


training_args = GRPOConfig(
    output_dir=output_dir,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
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
    reward_funcs=[sqa_reward_func, reason_reward_func]
)

print("Starting training...")
trainer.train()
trainer.save_model(output_dir)
print(f"Training completed.")
