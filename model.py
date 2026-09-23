"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer

def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # TODO: load and return the Hugging Face tokenizer for the given model name.
    return AutoTokenizer.from_pretrained(model_name)


# tok = load_distilgpt2_tokenizer()
# print(tok.decode(tok.encode("ji")))

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM

def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # ...ForCausalLM adds the LM head, so forward() returns logits over the vocab
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # eval() turns off dropout so two greedy decodes of the same prompt agree
    model.eval()
    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    # TODO: assign tokenizer.pad_token = tokenizer.eos_token and return the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    
    inputs = tokenizer(prompt, return_tensors="pt")        # dict of tensors, not a list of ints
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,                  # greedy
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)   # out is (1, T); decode the row

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    # TODO: return the token id with the largest logit as a Python int
    out= torch.argmax(logits)
    return out.item()

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    # TODO: rescale logits by temperature, softmax, and sample one token id
    if(temperature>0):
        probablities = torch.softmax(logits/temperature,axis=-1) #-> This ensures that logits are converted into probs and sum to 1.
        return torch.multinomial(probablities,1).item()

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # TODO: keep the k largest entries of logits and set the rest to -inf.
    if(k < logits.shape[0]):
        threshold = torch.topk(logits,k).values[-1]
        mask = (logits >=threshold)
        return torch.where(mask, logits, float('-inf'))
    else:
        return logits

# Step 8 - top_p_filter
import numpy as np

def top_p_filter(logits, p):
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max()
    probs = np.exp(z) / np.exp(z).sum()
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    keep_sorted = np.empty(len(order), dtype=bool)
    keep_sorted[0] = True
    keep_sorted[1:] = cum[:-1] < p
    out = np.full_like(logits, -np.inf)
    kept = order[keep_sorted]
    out[kept] = logits[kept]
    return out

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    return [
        {"prompt": "What is the capital of France?", "response": "Paris."},
        {"prompt": "Name a primary color.", "response": "Red."},
        {"prompt": "How many legs does a spider have?", "response": "Eight."},
        {"prompt": "Translate 'hello' to Spanish.", "response": "Hola."},
        {"prompt": "What is 2 + 2?", "response": "4."},
        {"prompt": "Which planet is closest to the Sun?", "response": "Mercury."},
        {"prompt": "How many r's are in 'strawberry'?", "response": "Three."},
    ]

# Step 10 - format_example
def format_example(example):
    # TODO: render {'prompt','response'} into one training string with role markers
    

 return f'### Instruction:\n{example["prompt"]}\n\n### Response:\n{example["response"]}'

# Step 11 - apply_template
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
    
    formatted_examples_list  = []
    for example in examples:

        formatted_examples_list.append(format_example(example))
    
    return formatted_examples_list

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]

    input_ids = tokenizer.encode(text, truncation = True , max_length= max_length, padding = False)
    return input_ids

# Step 13 - build_labels
import copy

def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels
    labels = copy.deepcopy(input_ids)

    return labels

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # TODO: replace the first prompt_length entries of labels with -100 and return the new list


    masked_labels = copy.deepcopy(labels)
    n = len(labels)
    
    if prompt_length < n:
        for idx in range(prompt_length):
            masked_labels[idx] = -100
        
        return masked_labels
    
    return [-100]*n

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    max_len = max(len(s) for s in sequences)
    return [s + [pad_id] * (max_len - len(s)) for s in sequences]

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    return [[int(t != pad_id) for t in row] for row in padded_ids]

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    ids = pad_batch([ex["input_ids"] for ex in batch], pad_id)
    labels = pad_batch([ex["labels"] for ex in batch], -100)
    mask = make_attention_mask(ids, pad_id)
    return {k: torch.tensor(v, dtype=torch.long)
            for k, v in (("input_ids", ids), ("labels", labels), ("attention_mask", mask))}

# Step 18 - iterate_minibatches
import random

def iterate_minibatches(examples, batch_size, seed=0):
    items = list(examples)                      # a copy, so the caller's list keeps its order
    random.Random(seed).shuffle(items)          # a local RNG, so the global random state isn't touched
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]           # the last slice holds whatever is left over

# Step 19 - train_val_split
import math
def train_val_split(examples, val_ratio=0.2, seed=0):
    # TODO: deterministically split examples into (train, val) using seed and val_ratio

    random.Random(seed).shuffle(examples)
    split = math.floor(len(examples) * val_ratio)
    val_spit = examples[:split]
    train_split = examples[split:]
    return train_split, val_spit

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()   # predictions at positions 0..T-2
    shift_labels = labels[:, 1:].contiguous()       # the tokens they should predict: 1..T-1
    return shift_logits, shift_labels

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    v = shift_logits.shape[-1]
    return F.cross_entropy(shift_logits.reshape(-1, v),   # (B*(T-1), V)
                           shift_labels.reshape(-1),      # (B*(T-1),)
                           ignore_index=-100)

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # TODO: initialize state on first call, then update moments and apply the decoupled AdamW step
    b1, b2 = betas
    if not state:
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)
    state["step"] += 1
    t, m, v = state["step"], state["m"], state["v"]
    with torch.no_grad():
        param.mul_(1 - lr * weight_decay)
        m.mul_(b1).add_(grad, alpha=1 - b1)
        v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        denom = (v / (1 - b2 ** t)).sqrt_().add_(eps)
        param.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
    return param

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    if warmup_steps <= 0 or step >= warmup_steps:
        return 1.0
    return max(0.0, step / warmup_steps)

# Step 24 - clip_grad_norm
import torch

def clip_grad_norm(grads, max_norm):
    total = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads)).item()
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor

    return torch.stack(grad_list).mean(dim=0)

# Step 26 - sft_train_step
import torch

def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # TODO: forward the batch, compute shifted cross-entropy loss, backprop, step optimizer
    optimizer.zero_grad()
    logits = model(input_ids = batch["input_ids"],attention_mask=batch["attention_mask"]).logits
    loss= cross_entropy_loss(*shift_logits_and_labels(logits, batch["labels"]))
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 27 - evaluate_loss
import torch

def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    # TODO: iterate batches under no_grad, shift logits/labels, average cross-entropy.
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in batches:
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            loss = cross_entropy_loss(*shift_logits_and_labels(logits, batch["labels"]))
            losses.append(loss.item())
        
    return sum(losses) / len(losses) if losses else float("nan")

# Step 28 - lora_delta
def lora_delta(A, B, alpha, r):
    # TODO: build the scaled low-rank weight update from factors A and B.
    return (alpha / r) * (B @ A)

# Step 29 - lora_linear_forward
def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    # TODO: return x @ (base_weight + lora_delta).T (+ bias) using lora_delta(A, B, alpha, r)
    

    if bias!= None:
        return x @ (base_weight + lora_delta(A,B, alpha, r)).T + bias

    return x @ (base_weight + lora_delta(A,B, alpha, r)).T

