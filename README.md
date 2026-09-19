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

---

Built on Deep-ML.
