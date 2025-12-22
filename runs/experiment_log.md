# Experiment Log

Run entries. Fill results and observations after each run.

- run_name: <string>
- date: <YYYY-MM-DD>
- dataset: <train_txt/train_tokens_npy, valid_txt/valid_tokens_npy>
- tokenizer: <gpt2|bpe>, special_tokens: <list>
- model: d_model=<int>, num_layers=<int>, num_heads=<int>, d_ff=<int>, context_length=<int>
- optimizer: AdamW lr=<float>, betas=(<float>,<float>), eps=<float>, weight_decay=<float>
- schedule: max_lr=<float>, min_lr=<float>, warmup_iters=<int>, cosine_cycle_iters=<int>
- batch_size: <int>
- grad_clip: <float>
- checkpoint_path: <path>
- wall_clock_start: <timestamp>
- metrics_file: runs/<run_name>/metrics.csv
- wandb: project=<name>, run_name=<name>
- training_curve_summary: <observations>
- validation_curve_summary: <observations>
- tokens_per_sec: <observations>
- decoding_examples: <prompt, settings, output>
- notes: <anything notable>

## Example Template

- run_name: baseline-tinystories
- date: 2025-12-15
- dataset: train_txt=data/TinyStoriesV2-GPT4-train.txt, valid_txt=data/TinyStoriesV2-GPT4-valid.txt
- tokenizer: bpe, special_tokens=["<|endoftext|>"]
- model: d_model=512, num_layers=6, num_heads=8, d_ff=2048, context_length=512
- optimizer: AdamW lr=3e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01
- schedule: max_lr=3e-4, min_lr=3e-5, warmup_iters=400, cosine_cycle_iters=10000
- batch_size: 32
- grad_clip: 1.0
- checkpoint_path: runs/ckpt.pt
- wall_clock_start: <auto>
- metrics_file: runs/baseline-tinystories/metrics.csv
- wandb: project=cs336-hw1, run_name=baseline-tinystories
- training_curve_summary: TBD
- validation_curve_summary: TBD
- tokens_per_sec: TBD
- decoding_examples:
  - prompt: "Once upon a time,"
    settings: temperature=0.8, top_p=0.9, max_new_tokens=100
    output: TBD
- notes: TBD
