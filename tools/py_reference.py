"""
Python Reference Implementation of LeagueOfLLMs Inference Engine
Ini adalah implementasi referensi 1-ke-1 dari logika C++ kita di include/engine.hpp.
Dapat dijalankan langsung dengan Python murni + numpy untuk membuktikan kebenaran
arsitektur Transformer, KV-Cache, dan BPE Tokenizer!
"""

import struct
import numpy as np
import math
import sys
import os

class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len

class Tokenizer:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.max_token_len = struct.unpack("i", f.read(4))[0]
            self.vocab = []
            self.token_to_id = {}
            for i in range(32000):
                score = struct.unpack("f", f.read(4))[0]
                length = struct.unpack("i", f.read(4))[0]
                piece = f.read(length).decode("utf-8", errors="replace")
                self.vocab.append((piece, score))
                self.token_to_id[piece] = i

    def encode(self, text):
        tokens = [1] # BOS
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(ord(char) + 3)
        
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                merged = self.vocab[tokens[i]][0] + self.vocab[tokens[i+1]][0]
                if merged in self.token_to_id:
                    score = self.vocab[self.token_to_id[merged]][1]
                    if score > best_score:
                        best_score = score
                        best_id = self.token_to_id[merged]
                        best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]
        return tokens

    def decode(self, token_id):
        return self.vocab[token_id][0]

def rmsnorm(x, weight, eps=1e-5):
    mean_sq = np.mean(x ** 2)
    return (x / np.sqrt(mean_sq + eps)) * weight

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def rope(x, pos, head_dim):
    for i in range(0, head_dim, 2):
        freq = 1.0 / (10000.0 ** (i / head_dim))
        val = pos * freq
        cos_v, sin_v = math.cos(val), math.sin(val)
        v0, v1 = x[i], x[i+1]
        x[i] = v0 * cos_v - v1 * sin_v
        x[i+1] = v0 * sin_v + v1 * cos_v
    return x

class PyEngine:
    def __init__(self, model_path, tok_path):
        self.tokenizer = Tokenizer(tok_path)
        with open(model_path, "rb") as f:
            cfg_raw = struct.unpack("7i", f.read(28))
            self.cfg = Config(*cfg_raw)
            weights_data = np.frombuffer(f.read(), dtype=np.float32)

        # Parse weights
        c = self.cfg
        ptr = 0
        self.embed = weights_data[ptr : ptr + c.vocab_size * c.dim].reshape(c.vocab_size, c.dim)
        ptr += c.vocab_size * c.dim

        L, d, hd = c.n_layers, c.dim, c.hidden_dim
        head_size = d // c.n_heads
        kv_dim = c.n_kv_heads * head_size

        self.rms_att = []
        for _ in range(L): self.rms_att.append(weights_data[ptr : ptr + d]); ptr += d

        self.wq = []
        for _ in range(L): self.wq.append(weights_data[ptr : ptr + d * d].reshape(d, d)); ptr += d * d

        self.wk = []
        for _ in range(L): self.wk.append(weights_data[ptr : ptr + kv_dim * d].reshape(kv_dim, d)); ptr += kv_dim * d

        self.wv = []
        for _ in range(L): self.wv.append(weights_data[ptr : ptr + kv_dim * d].reshape(kv_dim, d)); ptr += kv_dim * d

        self.wo = []
        for _ in range(L): self.wo.append(weights_data[ptr : ptr + d * d].reshape(d, d)); ptr += d * d

        self.rms_ffn = []
        for _ in range(L): self.rms_ffn.append(weights_data[ptr : ptr + d]); ptr += d

        self.w1 = []
        for _ in range(L): self.w1.append(weights_data[ptr : ptr + hd * d].reshape(hd, d)); ptr += hd * d

        self.w2 = []
        for _ in range(L): self.w2.append(weights_data[ptr : ptr + d * hd].reshape(d, hd)); ptr += d * hd

        self.w3 = []
        for _ in range(L): self.w3.append(weights_data[ptr : ptr + hd * d].reshape(hd, d)); ptr += hd * d

        self.rms_final = weights_data[ptr : ptr + d]; ptr += d
        self.wcls = self.embed # Shared weights

        # KV Cache: [L, seq_len, kv_dim]
        self.key_cache = np.zeros((L, c.seq_len, kv_dim), dtype=np.float32)
        self.val_cache = np.zeros((L, c.seq_len, kv_dim), dtype=np.float32)

    def forward(self, token, pos):
        c = self.cfg
        x = self.embed[token].copy()
        head_size = c.dim // c.n_heads
        kv_dim = c.n_kv_heads * head_size

        for l in range(c.n_layers):
            # Attention RMSNorm
            xb = rmsnorm(x, self.rms_att[l])

            # QKV Projections
            q = np.dot(self.wq[l], xb)
            k = np.dot(self.wk[l], xb)
            v = np.dot(self.wv[l], xb)

            # RoPE
            for h in range(c.n_heads):
                q[h*head_size : (h+1)*head_size] = rope(q[h*head_size : (h+1)*head_size], pos, head_size)
            for h in range(c.n_kv_heads):
                k[h*head_size : (h+1)*head_size] = rope(k[h*head_size : (h+1)*head_size], pos, head_size)

            # KV-Cache save
            self.key_cache[l, pos] = k
            self.val_cache[l, pos] = v

            # Multi-Head Attention
            out_att = np.zeros(c.dim, dtype=np.float32)
            scale = 1.0 / math.sqrt(head_size)
            for h in range(c.n_heads):
                q_h = q[h*head_size : (h+1)*head_size]
                # Cached K up to pos
                k_cached = self.key_cache[l, : pos + 1, h*head_size : (h+1)*head_size]
                scores = np.dot(k_cached, q_h) * scale
                weights = softmax(scores)

                v_cached = self.val_cache[l, : pos + 1, h*head_size : (h+1)*head_size]
                out_att[h*head_size : (h+1)*head_size] = np.dot(weights, v_cached)

            # Out projection + Residual
            x = x + np.dot(self.wo[l], out_att)

            # FFN Block: SwiGLU
            xb = rmsnorm(x, self.rms_ffn[l])
            h1 = np.dot(self.w1[l], xb)
            h3 = np.dot(self.w3[l], xb)
            silu_h1 = h1 / (1.0 + np.exp(-h1))
            ffn_out = np.dot(self.w2[l], silu_h1 * h3)
            x = x + ffn_out

        x = rmsnorm(x, self.rms_final)
        logits = np.dot(self.wcls, x)
        return logits

    def generate(self, prompt, max_tokens=32, temperature=0.8):
        tokens = self.tokenizer.encode(prompt)
        print(f"\n[Prompt]: {prompt}", end="", flush=True)
        pos = 0
        token = tokens[0]
        prompt_len = len(tokens)

        for _ in range(max_tokens):
            logits = self.forward(token, pos)
            if pos + 1 < prompt_len:
                next_tok = tokens[pos + 1]
            else:
                probs = softmax(logits / temperature)
                next_tok = int(np.random.choice(len(probs), p=probs))
                if next_tok == 2: # EOS
                    break
                print(self.tokenizer.decode(next_tok), end="", flush=True)
            token = next_tok
            pos += 1
        print("\n")

if __name__ == "__main__":
    prompt = "Once upon a time, there was a little robot"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    engine = PyEngine("models/stories15M.bin", "models/tokenizer.bin")
    engine.generate(prompt, max_tokens=35, temperature=0.7)
