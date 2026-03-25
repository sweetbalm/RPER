import os
import re
import json
import torch
from PIL import Image
from datasets import Dataset


def build_prompt_from_sample(sample):
    question = sample["question"]
    choices = sample["options"]
    options = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
    prompt_text = (
        f"{question}\nOptions:\n{options}\n"
        "Please reason step by step. "
        "Rules for reasoning:\n"
        "1. Be extremely concise and use telegraphic style (omit non-essential words).\n"
        "2. DO NOT describe the image content unless necessary.\n"
        "3. DO NOT restate the question.\n"
        "After reasoning, provide the final answer in the format: 'So the answer is X.'."
    )
    return prompt_text


def get_one_sample(data_dir="/storage/hujiacong/gh/datasets/mathvision", split="test.json"):
    with open(os.path.join(data_dir, split)) as f:
        problems = json.load(f)

    p = problems['854']
    image_path = os.path.join(data_dir, p['image'])
    if not os.path.exists(image_path):
        print(f"{image_path} does not exist")
    img = Image.open(image_path).convert("RGB")
    prompt_text = build_prompt_from_sample(p)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    sample = {"messages": conversation, "answer": p["answer"], "image": img}
    return sample


def load_dataset(data_dir="/storage/hujiacong/gh/datasets/mathvision", split="train.json", num_samples=None):
    with open(os.path.join(data_dir, split)) as f:
        ps = json.load(f)

    samples = []
    for pid, p in ps.items():
        if num_samples is not None and len(samples) >= num_samples:
            break
        
        image_path = os.path.join(data_dir, p['image'])
        if not os.path.exists(image_path):
            print(f"{image_path} does not exist")
            continue
        image = Image.open(image_path).convert("RGB")
        prompt_text = build_prompt_from_sample(p)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        samples.append({
            "prompt": conversation,
            "images": [image],
            "answer": p["answer"]
        })
    return Dataset.from_list(samples)

def load_test(data_dir="/storage/hujiacong/gh/datasets/mathvision", split="test.json", num_samples=None):
    with open(os.path.join(data_dir, split)) as f:
        ps = json.load(f)

    samples = []
    for pid, p in ps.items():
        if num_samples is not None and len(samples) >= num_samples:
            break
        
        image_path = os.path.join(data_dir, p['image'])
        if not os.path.exists(image_path):
            print(f"{image_path} does not exist")
            continue
        img = Image.open(image_path).convert("RGB")
        prompt_text = build_prompt_from_sample(p)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        samples.append({
            "messages": conversation,
            "answer": p["answer"]
        })
    return samples


def parse_answer(response) -> str:
    # 查找最后一个独立出现的大写字母（A-D）
    matches = re.findall(r'\b[A-D]\b', response)
    if matches:
        return matches[-1]
    return ""


def sqa_reward_func(completions, answer, **kwargs):
    rewards = []
    for comp, ans in zip(completions, answer):
        if isinstance(comp, list) and len(comp) > 0:
            content = comp[0].get("content", "")
        else:
            content = ""
        reward = 0.0
        matches = re.findall(r'\b([A-D])\b', content)
        if matches and matches[-1] == ans:
            reward = 1.0
        rewards.append(reward)
    return rewards


def reason_reward_func(completions, **kwargs):
    rewards = []
    pattern_answer = r"So the answer is [A-Z]\.$"

    for comp in completions:
        content = ""
        if isinstance(comp, str):
            content = comp
        elif isinstance(comp, list) and len(comp) > 0:
            content = comp[0].get("content", "")

        match_answer = re.search(pattern_answer, content.strip())

        score = 0.0

        if len(content) > 30:
            score += 0.5
        else:
            score -= 0.5

        if match_answer:
            score += 0.5
        else:
            score -= 0.5

        rewards.append(score)

    return rewards


def compute_head_avg_backward_distance(attentions, response_positions, device="cuda"):
    L = len(attentions)
    H = attentions[0].shape[1]
    N = attentions[0].shape[2]
    R = torch.tensor(response_positions, device=device)

    t_indices = torch.arange(N, device=device).unsqueeze(1)
    s_indices = torch.arange(N, device=device).unsqueeze(0)
    distance_matrix = t_indices - s_indices

    d_lh = torch.zeros(L, H, device=device)

    for l in range(L):
        A_l = attentions[l].squeeze(0)
        A_l_R = A_l[:, R, :]
        dist_R = distance_matrix[R, :]

        weighted_dist = A_l_R * dist_R
        sum_weighted_dist = weighted_dist.sum(dim=-1)

        d_lh[l] = sum_weighted_dist.mean(dim=1)

    return d_lh


def group_heads_by_span(d_lh, local_ratio=0.3, global_ratio=0.3):
    L, H = d_lh.shape
    all_d_vals = d_lh.view(-1)

    sorted_vals, sorted_idx = torch.sort(all_d_vals)

    n_heads = L * H
    n_local = int(n_heads * local_ratio)
    n_global = int(n_heads * global_ratio)

    local_flat_idx = sorted_idx[:n_local]
    global_flat_idx = sorted_idx[-n_global:]

    local_heads = [(int(idx // H), int(idx % H)) for idx in local_flat_idx]
    global_heads = [(int(idx // H), int(idx % H)) for idx in global_flat_idx]

    return local_heads, global_heads


def compute_waad(A_loc, response_positions, W=10):
    if A_loc.dim() == 3:
        A_loc = A_loc.squeeze(0)

    N = A_loc.shape[0]
    device = A_loc.device

    indices = torch.arange(N, device=device)
    dist_matrix = indices.unsqueeze(1) - indices.unsqueeze(0)

    clipped_dist = torch.clamp(dist_matrix, min=0, max=W)

    waad_all = (A_loc * clipped_dist).sum(dim=-1)  # [N]

    if isinstance(response_positions, list):
        response_positions = torch.tensor(response_positions, device=device)

    return waad_all[response_positions]


def compute_fai(A_glob, response_positions, H_lo=10, H_hi=50):
    if A_glob.dim() == 3:
        A_glob = A_glob.squeeze(0)

    N = A_glob.shape[0]
    device = A_glob.device

    t_indices = torch.arange(N, device=device).unsqueeze(1)  # [N, 1]
    s_indices = torch.arange(N, device=device).unsqueeze(0)  # [1, N]

    distance = t_indices - s_indices
    dist_mask = (distance >= H_lo) & (distance <= H_hi)

    is_response = torch.zeros(N, device=device, dtype=torch.bool)
    if isinstance(response_positions, list):
        response_positions = torch.tensor(response_positions, device=device)
    is_response[response_positions] = True

    valid_mask = dist_mask & is_response.unsqueeze(1)

    masked_A = A_glob * valid_mask.float()

    sum_attention = masked_A.sum(dim=0)
    count_valid = valid_mask.sum(dim=0)

    fai_all = torch.zeros_like(sum_attention)
    mask_nonzero = count_valid > 0
    fai_all[mask_nonzero] = sum_attention[mask_nonzero] / count_valid[mask_nonzero]

    return fai_all[response_positions]


def compute_token_entropy(logits, response_positions):
    logits = logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy_full = -(probs * log_probs).sum(dim=-1)
    return entropy_full[response_positions]


def get_vision_token_range(input_ids):
    vision_start_id = 151652
    vision_end_id = 151653
    
    pmt = input_ids[0]
    
    start_indices = torch.where(pmt == vision_start_id)[0]
    end_indices = torch.where(pmt == vision_end_id)[0]
    
    if len(start_indices) == 0 or len(end_indices) == 0:
        raise ValueError("Could not find vision start/end tokens in the input_ids")
    
    v_start = start_indices[0].item()
    v_end = end_indices[0].item()
    
    return v_start, v_end
