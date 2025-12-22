import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cs336_basics.embedding import Embedding
from cs336_basics.rms_layer_norm import RMSLayerNorm
from cs336_basics.multi_head_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.linear import Linear
from cs336_basics.data_loading import load_data
from cs336_basics.adamw import AdamW
from cs336_basics.cosine_lr_scheduler import cosine_lr_scheduler
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.experiment_logger import ExperimentLogger


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, rope_theta: float | None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSLayerNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope_theta=rope_theta, max_seq_len=context_length, device=device)
        self.ln2 = RMSLayerNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(torch.empty(0), torch.empty(0), torch.empty(0), d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
        rope_theta: float | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, context_length=context_length, rope_theta=rope_theta, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSLayerNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


def load_tokens_memmap(path: str):
    arr = np.load(path, mmap_mode="r")
    return arr


def parse_special_tokens(s: str) -> list[str]:
    if not s:
        return []
    return [t for t in (x.strip() for x in s.split(",")) if t]


def build_tokens_npy(txt_path: str, out_npy_path: str, tokenizer_type: str, vocab_path: str | None, merges_path: str | None, special_tokens_str: str):
    special_tokens = parse_special_tokens(special_tokens_str)
    if tokenizer_type == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        allowed = set(special_tokens) if special_tokens else set()
        total = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                total += len(enc.encode(line, allowed_special=allowed))
        mm = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype=np.int32, shape=(total,))
        i = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                ids = enc.encode(line, allowed_special=allowed)
                n = len(ids)
                if n:
                    mm[i:i+n] = np.asarray(ids, dtype=np.int32)
                    i += n
        mm.flush()
    else:
        from cs336_basics.bpe_tokenizer import Tokenizer
        tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
        total = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                total += len(tok.encode(line))
        mm = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype=np.int32, shape=(total,))
        i = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                ids = tok.encode(line)
                n = len(ids)
                if n:
                    mm[i:i+n] = np.asarray(ids, dtype=np.int32)
                    i += n
        mm.flush()


def evaluate(model: nn.Module, valid_tokens: np.ndarray, batch_size: int, context_length: int, device: str, eval_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = load_data(valid_tokens, batch_size, context_length, device)
            x = x.long()
            y = y.long()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tokens_npy", type=str, default="")
    parser.add_argument("--valid_tokens_npy", type=str, default="")
    parser.add_argument("--train_txt", type=str, default="")
    parser.add_argument("--valid_txt", type=str, default="")
    parser.add_argument("--tokenizer_type", type=str, choices=["gpt2", "bpe"], default="gpt2")
    parser.add_argument("--vocab_path", type=str, default="")
    parser.add_argument("--merges_path", type=str, default="")
    parser.add_argument("--special_tokens", type=str, default="<|endoftext|>")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=200)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--rebuild_tokens", action="store_true")
    parser.add_argument("--decode_prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--eot_token", type=str, default="<|endoftext|>")
    args = parser.parse_args()

    device = args.device
    if not args.decode_prompt:
        train_tokens_path = args.train_tokens_npy or ("data/train_tokens.npy" if args.train_txt else "")
        valid_tokens_path = args.valid_tokens_npy or ("data/valid_tokens.npy" if args.valid_txt else "")
        if not train_tokens_path:
            raise SystemExit("Provide --train_tokens_npy or --train_txt")
        if not valid_tokens_path:
            raise SystemExit("Provide --valid_tokens_npy or --valid_txt")
        if args.rebuild_tokens:
            if not args.train_txt or not args.valid_txt:
                raise SystemExit("--rebuild_tokens requires --train_txt and --valid_txt")
            os.makedirs(os.path.dirname(train_tokens_path), exist_ok=True)
            os.makedirs(os.path.dirname(valid_tokens_path), exist_ok=True)
            build_tokens_npy(args.train_txt, train_tokens_path, args.tokenizer_type, args.vocab_path or None, args.merges_path or None, args.special_tokens)
            build_tokens_npy(args.valid_txt, valid_tokens_path, args.tokenizer_type, args.vocab_path or None, args.merges_path or None, args.special_tokens)
        else:
            if not os.path.exists(train_tokens_path):
                if not args.train_txt:
                    raise SystemExit("Missing --train_txt to build train tokens")
                os.makedirs(os.path.dirname(train_tokens_path), exist_ok=True)
                build_tokens_npy(args.train_txt, train_tokens_path, args.tokenizer_type, args.vocab_path or None, args.merges_path or None, args.special_tokens)
            if not os.path.exists(valid_tokens_path):
                if not args.valid_txt:
                    raise SystemExit("Missing --valid_txt to build valid tokens")
                os.makedirs(os.path.dirname(valid_tokens_path), exist_ok=True)
                build_tokens_npy(args.valid_txt, valid_tokens_path, args.tokenizer_type, args.vocab_path or None, args.merges_path or None, args.special_tokens)
        train_tokens = load_tokens_memmap(train_tokens_path)
        valid_tokens = load_tokens_memmap(valid_tokens_path)
        if train_tokens.shape[0] <= args.context_length:
            raise SystemExit(f"Training tokens ({train_tokens.shape[0]}) <= context_length ({args.context_length}). Pass --rebuild_tokens to regenerate tokens or reduce --context_length.")
        if valid_tokens.shape[0] <= args.context_length:
            raise SystemExit(f"Validation tokens ({valid_tokens.shape[0]}) <= context_length ({args.context_length}). Pass --rebuild_tokens to regenerate tokens or reduce --context_length.")

    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=torch.float32,
        rope_theta=args.rope_theta,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_it = 0
    if args.resume_path and os.path.exists(args.resume_path):
        start_it = load_checkpoint(args.resume_path, model, optimizer)

    if args.decode_prompt:
        if args.tokenizer_type == "gpt2":
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            allowed = set(parse_special_tokens(args.special_tokens))
            prompt_ids = enc.encode(args.decode_prompt, allowed_special=allowed)
            eot_id = enc.encode(args.eot_token, allowed_special=allowed)[0]
            ids = list(prompt_ids)
            model.eval()
            with torch.no_grad():
                for _ in range(args.max_new_tokens):
                    inp = torch.tensor(ids[-args.context_length :], dtype=torch.long, device=device).unsqueeze(0)
                    logits = model(inp)
                    next_logits = logits[0, -1, :]
                    if args.temperature <= 0:
                        next_id = int(torch.argmax(next_logits).item())
                    else:
                        scaled = next_logits / args.temperature
                        probs = torch.softmax(scaled, dim=-1)
                        if args.top_p < 1.0:
                            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                            cum = torch.cumsum(sorted_probs, dim=-1)
                            mask = cum <= args.top_p
                            if not torch.any(mask):
                                mask[0] = True
                            filt_idx = sorted_idx[mask]
                            filt_probs = sorted_probs[mask]
                            total = torch.sum(filt_probs)
                            if total <= 0 or torch.isnan(total):
                                next_id = int(sorted_idx[0].item())
                            else:
                                filt_probs = filt_probs / total
                                choice = torch.multinomial(filt_probs, 1)
                                next_id = int(filt_idx[choice].item())
                        else:
                            next_id = int(torch.multinomial(probs, 1).item())
                    ids.append(next_id)
                    if next_id == eot_id:
                        break
            text = enc.decode(ids)
            print(text)
            return
        else:
            from cs336_basics.bpe_tokenizer import Tokenizer
            tok = Tokenizer.from_files(args.vocab_path, args.merges_path, parse_special_tokens(args.special_tokens))
            prompt_ids = tok.encode(args.decode_prompt)
            eot_id = tok.encode(args.eot_token)[0]
            ids = list(prompt_ids)
            model.eval()
            with torch.no_grad():
                for _ in range(args.max_new_tokens):
                    inp = torch.tensor(ids[-args.context_length :], dtype=torch.long, device=device).unsqueeze(0)
                    logits = model(inp)
                    next_logits = logits[0, -1, :]
                    if args.temperature <= 0:
                        next_id = int(torch.argmax(next_logits).item())
                    else:
                        scaled = next_logits / args.temperature
                        probs = torch.softmax(scaled, dim=-1)
                        if args.top_p < 1.0:
                            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                            cum = torch.cumsum(sorted_probs, dim=-1)
                            mask = cum <= args.top_p
                            if not torch.any(mask):
                                mask[0] = True
                            filt_idx = sorted_idx[mask]
                            filt_probs = sorted_probs[mask]
                            total = torch.sum(filt_probs)
                            if total <= 0 or torch.isnan(total):
                                next_id = int(sorted_idx[0].item())
                            else:
                                filt_probs = filt_probs / total
                                choice = torch.multinomial(filt_probs, 1)
                                next_id = int(filt_idx[choice].item())
                        else:
                            next_id = int(torch.multinomial(probs, 1).item())
                    ids.append(next_id)
                    if next_id == eot_id:
                        break
            text = tok.decode(ids)
            print(text)
            return

    use_wandb = bool(args.wandb_project)
    logger = ExperimentLogger(log_dir=args.log_dir, run_name=(args.run_name or args.wandb_run_name or None), use_wandb=use_wandb, config=vars(args), wandb_project=args.wandb_project)

    model.train()
    t0 = time.time()
    for it in range(start_it, args.max_iters):
        x, y = load_data(train_tokens, args.batch_size, args.context_length, device)
        x = x.long()
        y = y.long()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            from cs336_basics.gradient_clipping import gradient_clipping
            gradient_clipping(model.parameters(), args.grad_clip)

        lr = cosine_lr_scheduler(it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()

        if (it + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            msg = f"iter {it+1} loss {loss.item():.4f} lr {lr:.6f} tokens/s {(args.batch_size*args.context_length*args.log_interval)/max(elapsed,1e-9):.0f}"
            print(msg)
            tokens_per_sec = (args.batch_size * args.context_length * args.log_interval) / max(elapsed, 1e-9)
            logger.log_train(step=it + 1, train_loss=loss.item(), lr=lr, tokens_per_sec=tokens_per_sec)
            t0 = time.time()

        if (it + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, valid_tokens, args.batch_size, args.context_length, device, args.eval_batches)
            print(f"eval iter {it+1} val_loss {val_loss:.4f}")
            logger.log_eval(step=it + 1, val_loss=val_loss)

        if (it + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)

    logger.close()


if __name__ == "__main__":
    main()
