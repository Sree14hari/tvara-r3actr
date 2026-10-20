"""
Model architecture and KV-cache inference for Tvara.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TvaraConfig:
    vocab_size: int = 16384
    dim: int = 768
    n_layers: int = 14
    n_heads: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 512
    multiple_of: int = 64
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        self.k = torch.zeros((max_batch_size, n_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.v = torch.zeros((max_batch_size, n_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.curr_pos = 0

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor):
        s = k_val.shape[2]
        self.k[:, :, self.curr_pos : self.curr_pos + s, :] = k_val
        self.v[:, :, self.curr_pos : self.curr_pos + s, :] = v_val
        k_out = self.k[:, :, : self.curr_pos + s, :]
        v_out = self.v[:, :, : self.curr_pos + s, :]
        self.curr_pos += s
        return k_out, v_out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 64):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Attention(nn.Module):
    def __init__(self, config: TvaraConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.num_rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        b, s, _ = x.shape
        xq = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        xk = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv)

        if self.num_rep > 1:
            xk = xk.repeat_interleave(self.num_rep, dim=1)
            xv = xv.repeat_interleave(self.num_rep, dim=1)

        is_causal = (s > 1)
        out = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, config: TvaraConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = SwiGLU(config.dim, multiple_of=config.multiple_of)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), freqs_cis, kv_cache=kv_cache)
        return h + self.ffn(self.ffn_norm(h))

class TvaraLM(nn.Module):
    def __init__(self, config: TvaraConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        head_dim = config.dim // config.n_heads
        freqs_cis = precompute_rope_freqs(head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis_real", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_cis_imag", freqs_cis.imag.float(), persistent=False)

    def forward(self, idx: torch.Tensor, start_pos: int = 0, kv_caches: Optional[List[KVCache]] = None):
        b, s = idx.shape
        h = self.tok_embeddings(idx)
        freqs_cis = torch.complex(
            self.freqs_cis_real[start_pos : start_pos + s],
            self.freqs_cis_imag[start_pos : start_pos + s]
        )

        for i, layer in enumerate(self.layers):
            c = kv_caches[i] if kv_caches is not None else None
            h = layer(h, freqs_cis, kv_cache=c)

        h = self.norm(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate_cached(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 220,
        temperature: float = 0.4,
        top_k: int = 40,
        repetition_penalty: float = 1.12,
        stop_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        stop_ids = stop_ids or []
        b, s = idx.shape
        head_dim = self.config.dim // self.config.n_heads
        amp_dtype = self.tok_embeddings.weight.dtype
        kv_caches = [
            KVCache(b, self.config.max_seq_len, self.config.n_kv_heads, head_dim, device=idx.device, dtype=amp_dtype)
            for _ in range(self.config.n_layers)
        ]

        # Prefill prompt
        _ = self(idx, start_pos=0, kv_caches=kv_caches)
        curr_pos = s
        curr_token = idx[:, -1:]
        out_tokens = idx.clone()

        for _ in range(max_new_tokens):
            if curr_pos >= self.config.max_seq_len:
                break

            logits = self(curr_token, start_pos=curr_pos, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                for token_id in set(out_tokens[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            out_tokens = torch.cat((out_tokens, next_token), dim=1)
            curr_token = next_token
            curr_pos += 1

            if next_token.item() in stop_ids:
                break

        return out_tokens
