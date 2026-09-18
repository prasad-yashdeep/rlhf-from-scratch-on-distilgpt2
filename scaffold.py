"""
RLHF from Scratch on DistilGPT2 scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end RLHF-from-scratch demo on a tiny GPT-2: SFT -> reward modeling -> PPO/DPO -> chat."""
import numpy as np
import torch

from solution import (
    load_distilgpt2_tokenizer,
    load_distilgpt2_model,
    set_pad_token_to_eos,
    generate_and_decode,
    greedy_decode,
    sample_with_temperature,
    top_k_filter,
    top_p_filter,
    build_synthetic_instruction_dataset,
    format_example,
    apply_template,
    tokenize_example,
    build_labels,
    mask_prompt_labels,
    pad_batch,
    make_attention_mask,
    collate_lm_batch,
    iterate_minibatches,
    train_val_split,
    shift_logits_and_labels,
    cross_entropy_loss,
    adamw_update,
    linear_warmup_schedule,
    clip_grad_norm,
    accumulate_gradients,
    sft_train_step,
    evaluate_loss,
    lora_delta,
    lora_linear_forward,
    init_lora_weights,
    freeze_base_params,
    count_trainable_params,
    merge_lora,
    build_synthetic_preference_dataset,
    format_preference,
    reward_head_forward,
    pairwise_reward_loss,
    reward_bce_loss,
    pairwise_accuracy,
    reward_train_step,
    sequence_logprob,
    per_token_kl,
    compute_returns,
    gae_advantages,
    policy_ratio,
    clipped_surrogate,
    value_function_loss,
    entropy_bonus,
    ppo_loss,
    kl_penalized_reward,
    batch_sequence_logprob,
    dpo_ref_logratios,
    dpo_loss,
    ipo_loss,
    kto_loss,
    orpo_loss,
    simpo_loss,
    build_eval_prompt_set,
    generate_completions,
    score_with_reward,
    win_rate,
    stream_tokens,
    apply_stop_tokens,
    chat,
)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # 1) Tokenizer + base model
    tokenizer = load_distilgpt2_tokenizer()
    set_pad_token_to_eos(tokenizer)
    model = load_distilgpt2_model()
    pad_id = tokenizer.pad_token_id
    print(f"Loaded tiny-gpt2; vocab={len(tokenizer)}, pad==eos? {tokenizer.pad_token == tokenizer.eos_token}")

    # 2) Baseline generation (un-aligned)
    base_out = generate_and_decode(model, tokenizer, "Hello, how are you?", max_new_tokens=8)
    print("Base completion:", repr(base_out))

    # 3) Build + tokenize SFT data
    sft_data = build_synthetic_instruction_dataset()
    train_data, val_data = train_val_split(sft_data, val_ratio=0.25, seed=0)
    train_texts = apply_template(train_data)
    val_texts = apply_template(val_data)

    def make_batches(texts, bs=2):
        examples = []
        for t in texts:
            enc = tokenize_example(tokenizer, t, max_length=32)
            ids = enc["input_ids"] if isinstance(enc, dict) else enc
            labels = build_labels(ids)
            examples.append({"input_ids": ids, "labels": labels})
        for mb in iterate_minibatches(examples, batch_size=bs, seed=0):
            yield collate_lm_batch(mb, pad_id)

    # 4) Short SFT loop and watch loss drop
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses = []
    for step in range(3):
        for batch in make_batches(train_texts, bs=2):
            loss = sft_train_step(model, batch, optimizer)
            losses.append(float(loss))
    val_loss = evaluate_loss(model, list(make_batches(val_texts, bs=2)))
    print(f"SFT train losses: {[round(l, 3) for l in losses[:6]]}... val_loss={float(val_loss):.3f}")

    # 5) Reward model: train a tiny head on synthetic preferences
    pref_data = build_synthetic_preference_dataset(num_examples=6, seed=0)
    hidden = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    reward_head = torch.nn.Linear(hidden, 1)
    rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)

    class _HiddenBackbone:
        """Adapter: the reward_train_step contract expects a callable returning a
        hidden-state tensor of shape (B, T, H), but we have a full LM head model."""
        def __init__(self, m):
            self.m = m
        def __call__(self, ids, attention_mask=None):
            out = self.m(ids, attention_mask=attention_mask, output_hidden_states=True)
            return out.hidden_states[-1]

    backbone = _HiddenBackbone(model)

    def _build_pref_batch(pref, tok, max_length=32):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        chosen_texts = [ex["prompt"] + " " + ex["chosen"] for ex in pref]
        rejected_texts = [ex["prompt"] + " " + ex["rejected"] for ex in pref]
        ce = tok(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        re_ = tok(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return {
            "chosen_input_ids": ce["input_ids"],
            "chosen_attention_mask": ce["attention_mask"],
            "rejected_input_ids": re_["input_ids"],
            "rejected_attention_mask": re_["attention_mask"],
        }

    rm_batch = _build_pref_batch(pref_data, tokenizer)
    rm_out = None
    for _ in range(2):
        rm_out = reward_train_step(backbone, reward_head, rm_batch, rm_opt)
    rm_loss = rm_out["loss"] if isinstance(rm_out, dict) else float(rm_out)
    print(f"Reward head trained; final RM loss ~ {float(rm_loss):.3f}")

    # 6) Compare aligned vs base via reward-model win-rate
    eval_prompts = build_eval_prompt_set()[:3]
    comps_aligned = generate_completions(model, tokenizer, eval_prompts, max_new_tokens=8)
    base_again = load_distilgpt2_model()
    comps_base = generate_completions(base_again, tokenizer, eval_prompts, max_new_tokens=8)

    # score_with_reward expects a dict bundle, not a bare nn.Linear.
    reward_bundle = {
        "model": model,
        "weight": reward_head.weight,
        "bias": reward_head.bias,
    }
    scored = [
        (score_with_reward(reward_bundle, tokenizer, p, c_a),
         score_with_reward(reward_bundle, tokenizer, p, c_b))
        for p, c_a, c_b in zip(eval_prompts, comps_aligned, comps_base)
    ]
    wins = sum(1 for a, b in scored if float(a) > float(b))
    print(f"Aligned beats base on {wins}/{len(scored)} prompts (reward-model judged)")

    # 7) Minimal chat interface
    reply = chat(model, tokenizer, "Say hi.", system_prompt="You are helpful.", max_new_tokens=8)
    print("Chat reply:", repr(reply))
