# RLHF from Scratch on DistilGPT2

Build the full Reinforcement Learning from Human Feedback pipeline on distilgpt2 from scratch: decoding, supervised fine-tuning, LoRA adapters, reward modeling, PPO, and preference-optimization methods like DPO, IPO, KTO, ORPO, and SimPO. Ends with evaluation tooling and a minimal chat interface to compare aligned and unaligned models.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_distilgpt2_tokenizer
- [x] **2.** load_distilgpt2_model
- [x] **3.** set_pad_token_to_eos
- [x] **4.** generate_and_decode
- [x] **5.** greedy_decode
- [x] **6.** sample_with_temperature
- [x] **7.** top_k_filter
- [x] **8.** top_p_filter
- [x] **9.** build_synthetic_instruction_dataset
- [x] **10.** format_example
- [x] **11.** apply_template
- [x] **12.** tokenize_example
- [x] **13.** build_labels
- [x] **14.** mask_prompt_labels
- [x] **15.** pad_batch
- [x] **16.** make_attention_mask
- [x] **17.** collate_lm_batch
- [x] **18.** iterate_minibatches
- [x] **19.** train_val_split
- [x] **20.** shift_logits_and_labels
- [x] **21.** cross_entropy_loss
- [x] **22.** adamw_update
- [x] **23.** linear_warmup_schedule
- [x] **24.** clip_grad_norm
- [x] **25.** accumulate_gradients
- [x] **26.** sft_train_step

---

Built on Deep-ML.
