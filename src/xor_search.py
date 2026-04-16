#!/usr/bin/env python3
"""XOR Patch Search for Bonsai 8B — GCP/CLI version.

Usage:
    # Single domain search with default editing probes
    python xor_search.py --iterations 300

    # Search with calibrated probe file
    python xor_search.py --probes probes_math_gsm8k.json --iterations 300

    # Specify output and model paths
    python xor_search.py --model ./Bonsai-8B.gguf --probes probes.json --output patch.json --iterations 300

    # Iterative multi-domain pipeline
    python xor_search.py --pipeline --iterations 200

Requires: pip install gguf transformers huggingface_hub torch numpy
"""

import argparse, os, sys, json, time, math, random, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

DEFAULT_MODEL_REPO = 'prism-ml/Bonsai-8B-gguf'
DEFAULT_MODEL_FILE = 'Bonsai-8B.gguf'
DEFAULT_SEARCH_LAYERS = [1, 2, 3, 4, 34]
GROUP_SIZE = 128

# ─────────────────────────────────────────────────────────
# GGUF Q1_0 Patch
# ─────────────────────────────────────────────────────────

def patch_gguf_q1_0():
    """Register Q1_0 (type 41) with the gguf library."""
    import gguf
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    try:
        GGMLQuantizationType(41)
    except ValueError:
        obj = int.__new__(GGMLQuantizationType, 41)
        obj._name_ = 'Q1_0'; obj._value_ = 41
        GGMLQuantizationType._member_map_['Q1_0'] = obj
        GGMLQuantizationType._value2member_map_[41] = obj
    if GGMLQuantizationType(41) not in GGML_QUANT_SIZES:
        GGML_QUANT_SIZES[GGMLQuantizationType(41)] = (128, 18)

# ─────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────

def ensure_model(model_path):
    """Download model if not present.

    Infers HF repo and filename from the basename, so e.g. 'Bonsai-1.7B.gguf'
    resolves to 'prism-ml/Bonsai-1.7B-gguf'. Falls back to the 8B defaults.
    """
    if os.path.exists(model_path):
        log.info(f'Model found: {model_path}')
        return model_path
    filename = os.path.basename(model_path)
    if filename.endswith('.gguf') and filename.startswith('Bonsai-'):
        repo = f'prism-ml/{filename[:-5]}-gguf'
    else:
        repo = DEFAULT_MODEL_REPO
        filename = DEFAULT_MODEL_FILE
    log.info(f'Downloading {filename} from {repo}...')
    from huggingface_hub import hf_hub_download
    local_dir = os.path.dirname(model_path) or '.'
    hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    return model_path


def load_model(model_path, device):
    """Parse GGUF, load weights to CPU, return config dict."""
    patch_gguf_q1_0()
    from gguf import GGUFReader

    log.info('Parsing GGUF...')
    reader = GGUFReader(model_path)

    meta = {}
    for field in reader.fields:
        try:
            parts = reader.fields[field].parts
            val = parts[-1].tolist() if hasattr(parts[-1], 'tolist') else parts[-1]
            if isinstance(val, list) and len(val) == 1: val = val[0]
            meta[field] = val
        except: pass

    tmap = {t.name: t for t in reader.tensors}

    def _meta(*keys, default):
        for k in keys:
            if k in meta:
                return meta[k]
        return default

    cfg = {
        'n_layers':   _meta('llama.block_count',                 'qwen2.block_count',                 'qwen3.block_count',                 default=36),
        'n_heads':    _meta('llama.attention.head_count',        'qwen2.attention.head_count',        'qwen3.attention.head_count',        default=32),
        'n_kv_heads': _meta('llama.attention.head_count_kv',     'qwen2.attention.head_count_kv',     'qwen3.attention.head_count_kv',     default=8),
        'hidden_dim': _meta('llama.embedding_length',            'qwen2.embedding_length',            'qwen3.embedding_length',            default=4096),
        'rms_eps':    1e-6,
        'rope_theta': _meta('llama.rope.freq_base',              'qwen2.rope.freq_base',              'qwen3.rope.freq_base',              default=1000000.0),
    }
    # Some Qwen3 GGUFs encode head_dim explicitly (necessary for 1.7B where
    # head_dim != hidden_dim/n_heads).
    explicit_head_dim = _meta('llama.attention.key_length',
                              'qwen2.attention.key_length',
                              'qwen3.attention.key_length',
                              default=None)
    cfg['head_dim'] = explicit_head_dim if explicit_head_dim else (cfg['hidden_dim'] // cfg['n_heads'])

    gate_name = [n for n in tmap if 'ffn_gate' in n and 'weight' in n][0]
    gs = tuple(int(s) for s in tmap[gate_name].shape)
    cfg['intermediate_dim'] = gs[1] if gs[0] == cfg['hidden_dim'] else gs[0]

    log.info(f"Arch: {cfg['n_layers']}L {cfg['hidden_dim']}D {cfg['intermediate_dim']}I "
             f"{cfg['n_heads']}/{cfg['n_kv_heads']} heads rope={cfg['rope_theta']}")

    log.info('Loading weights to CPU...')
    weights = {}
    for name, tensor in tmap.items():
        weights[name] = {
            'raw': torch.tensor(np.array(tensor.data)),
            'shape': tuple(int(s) for s in tensor.shape),
            # Normalize "GGMLQuantizationType.Q1_0" -> "Q1_0"
            'type': str(tensor.tensor_type).rsplit('.', 1)[-1],
        }
    log.info(f'{len(weights)} tensors loaded')

    return cfg, weights

# ─────────────────────────────────────────────────────────
# Dequantization
# ─────────────────────────────────────────────────────────

class BonsaiEngine:
    """Manages dequantization, forward pass, and bit manipulation."""

    # Tensor naming
    EMBED = 'token_embd.weight'; NORM = 'output_norm.weight'; LM_HEAD = 'output.weight'
    ATTN_Q = 'attn_q.weight'; ATTN_K = 'attn_k.weight'
    ATTN_V = 'attn_v.weight'; ATTN_O = 'attn_output.weight'
    FFN_GATE = 'ffn_gate.weight'; FFN_UP = 'ffn_up.weight'; FFN_DOWN = 'ffn_down.weight'
    ATTN_NORM = 'attn_norm.weight'; FFN_NORM = 'ffn_norm.weight'
    PROJ_LIST = [FFN_GATE, FFN_UP, FFN_DOWN]

    def __init__(self, cfg, weights, device, search_layers=None):
        self.cfg = cfg
        self.weights = weights
        self.device = device
        self.search_layers = search_layers or DEFAULT_SEARCH_LAYERS
        self._shifts = torch.arange(32, device=device, dtype=torch.int32)
        self._dequant_cache = {}
        self._unpacked_cache = {}
        self.tokenizer = None
        # Auto-enable weight caching if VRAM > 40 GB (A100/H100).
        # Caches dequantized Q1_0 weights on GPU, eliminating per-call
        # dequantization.  Requires ~25 GB for Qwen3-8B.
        # Auto-enable weight caching when the dequantized model fits in VRAM
        # with room for activations. Q1_0 stores each weight in ~1.125 bits;
        # dequantization to fp16 expands ~16×. So estimate dequant footprint
        # from total weight-param count × 2 bytes. Cache if that plus ~5 GB
        # working headroom fits in the GPU.
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9 if device.type == 'cuda' else 0
        # Param count estimate from cfg (embed + per-layer attn/ffn).
        nL, hD, iD, nH, nKV, hdim = cfg['n_layers'], cfg['hidden_dim'], cfg['intermediate_dim'], cfg['n_heads'], cfg['n_kv_heads'], cfg['head_dim']
        VOCAB = 151936
        params = (
            VOCAB * hD                          # token_embd
            + nL * (hD * nH * hdim              # attn_q
                    + hD * nKV * hdim * 2       # attn_k + attn_v
                    + nH * hdim * hD            # attn_o
                    + hD * iD * 3)              # ffn_gate + ffn_up + ffn_down
        )
        dequant_gb = params * 2 / 1e9           # fp16
        headroom_gb = 5.0
        self.cache_dequant = (vram_gb - dequant_gb) > headroom_gb
        log.info(
            f'VRAM {vram_gb:.0f} GB, dequantized footprint est {dequant_gb:.1f} GB → '
            f'cache {"ENABLED" if self.cache_dequant else "disabled"}'
        )

    @staticmethod
    def layer_name(layer, suffix):
        return f'blk.{layer}.{suffix}'

    # ── Q1_0 unpacking ──

    def unpack_q1_0(self, raw_tensor, out_f, in_f):
        raw_bytes = raw_tensor.to(torch.uint8).numpy().ravel()
        ng = (int(out_f) * int(in_f)) // GROUP_SIZE
        bpg = (GROUP_SIZE // 8) + 2
        offsets = np.arange(int(ng)) * int(bpg)
        sr = np.zeros((ng, 2), dtype=np.uint8)
        sr[:, 0] = raw_bytes[offsets]; sr[:, 1] = raw_bytes[offsets + 1]
        scales = sr.view(np.float16).reshape(ng)
        br = np.zeros((ng, 16), dtype=np.uint8)
        for i in range(16): br[:, i] = raw_bytes[offsets + 2 + i]
        packed = br.view(np.uint32).reshape(ng, 4)
        # np.uint32 isn't directly supported by torch.tensor on numpy>=2.
        # View as int32 to preserve the bit pattern (bit-ops rely on pattern,
        # not value; `(x >> k) & 1` is correct regardless of signedness).
        packed_i32 = packed.view(np.int32)
        return (
            torch.tensor(packed_i32, dtype=torch.int32, device=self.device),
            torch.tensor(scales, dtype=torch.float16, device=self.device),
        )

    def dequantize(self, packed, scales, out_f, in_f):
        ng = packed.shape[0]
        total = int(out_f) * int(in_f)
        if total < 60_000_000:
            bits = ((packed.unsqueeze(-1) >> self._shifts) & 1).float().reshape(ng, GROUP_SIZE)
            return (scales.float().unsqueeze(1) * (2.0 * bits - 1.0)).reshape(int(out_f), int(in_f))
        chunks = []
        for s in range(0, ng, 50000):
            e = min(s + 50000, ng)
            b = ((packed[s:e].unsqueeze(-1) >> self._shifts) & 1).float().reshape(e-s, GROUP_SIZE)
            chunks.append((scales[s:e].float().unsqueeze(1) * (2.0 * b - 1.0)).reshape(-1))
            del b
        flat = torch.cat(chunks); del chunks
        return flat.reshape(int(out_f), int(in_f))

    def load_fp(self, name):
        w = self.weights[name]
        raw_bytes = w['raw'].numpy().view(np.uint8).tobytes()
        dt = np.float16 if w['type'] in ('1', 'F16', 'f16') else np.float32
        return torch.tensor(np.frombuffer(raw_bytes, dtype=dt).reshape(w['shape']),
                            dtype=torch.float32, device=self.device)

    def get_weight(self, name):
        if name in self._dequant_cache:
            return self._dequant_cache[name]
        if name not in self.weights:
            return None  # caller (e.g. tied LM head) handles fallback
        w = self.weights[name]
        if w['type'] in ('Q1_0', 'q1_0', '41'):
            out_f = w['shape'][1] if len(w['shape']) > 1 else 1
            in_f = w['shape'][0]
            if name not in self._unpacked_cache:
                packed, scales = self.unpack_q1_0(w['raw'], out_f, in_f)
                self._unpacked_cache[name] = (packed, scales, out_f, in_f)
            packed, scales, out_f, in_f = self._unpacked_cache[name]
            result = self.dequantize(packed, scales, out_f, in_f)
            if self.cache_dequant:
                self._dequant_cache[name] = result
            return result
        else:
            result = self.load_fp(name)
            self._dequant_cache[name] = result
            return result

    def linear(self, x, name):
        w = self.get_weight(name)
        result = x @ w.t()
        if not self.cache_dequant:
            del w; torch.cuda.empty_cache()
        return result

    # ── Forward pass ──

    @staticmethod
    def rms_norm(x, weight, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def build_rope(self, seq_len):
        hd = self.cfg['head_dim']
        theta = self.cfg['rope_theta']
        pos = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=self.device, dtype=torch.float32) / hd))
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([angles, angles], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def apply_rope(x, cos, sin):
        seq = x.shape[2]
        cos = cos[:seq].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq].unsqueeze(0).unsqueeze(0)
        d2 = x.shape[-1] // 2
        x_rot = torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)
        return x * cos + x_rot * sin

    @staticmethod
    def repeat_kv(x, n_rep):
        if n_rep == 1: return x
        b, nkv, s, hd = x.shape
        return x.unsqueeze(2).expand(b, nkv, n_rep, s, hd).reshape(b, nkv * n_rep, s, hd)

    @torch.no_grad()
    def forward(self, input_ids):
        cfg = self.cfg
        seq_len = input_ids.shape[1]
        rope_cos, rope_sin = self.build_rope(seq_len)
        n_rep = cfg['n_heads'] // cfg['n_kv_heads']
        ln = self.layer_name

        embed_w = self.get_weight(self.EMBED)
        h = embed_w[input_ids[0]].unsqueeze(0).float()
        if not self.cache_dequant:
            del embed_w; torch.cuda.empty_cache()

        for layer in range(cfg['n_layers']):
            normed = self.rms_norm(h, self.get_weight(ln(layer, self.ATTN_NORM)), cfg['rms_eps'])
            q = self.linear(normed, ln(layer, self.ATTN_Q))
            k = self.linear(normed, ln(layer, self.ATTN_K))
            v = self.linear(normed, ln(layer, self.ATTN_V))
            q = q.view(1, seq_len, cfg['n_heads'], cfg['head_dim']).transpose(1, 2)
            k = k.view(1, seq_len, cfg['n_kv_heads'], cfg['head_dim']).transpose(1, 2)
            v = v.view(1, seq_len, cfg['n_kv_heads'], cfg['head_dim']).transpose(1, 2)
            q = self.rms_norm(q, self.get_weight(ln(layer, 'attn_q_norm.weight')), cfg['rms_eps'])
            k = self.rms_norm(k, self.get_weight(ln(layer, 'attn_k_norm.weight')), cfg['rms_eps'])
            q = self.apply_rope(q, rope_cos, rope_sin)
            k = self.apply_rope(k, rope_cos, rope_sin)
            k = self.repeat_kv(k, n_rep); v = self.repeat_kv(v, n_rep)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(cfg['head_dim'])
            if seq_len > 1:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(1, seq_len, cfg['hidden_dim'])
            h = h + self.linear(attn_out, ln(layer, self.ATTN_O))
            normed_ffn = self.rms_norm(h, self.get_weight(ln(layer, self.FFN_NORM)), cfg['rms_eps'])
            gate = self.linear(normed_ffn, ln(layer, self.FFN_GATE))
            up = self.linear(normed_ffn, ln(layer, self.FFN_UP))
            h = h + self.linear(F.silu(gate) * up, ln(layer, self.FFN_DOWN))
            del q, k, v, attn, attn_out, gate, up, normed, normed_ffn
            if not self.cache_dequant:
                torch.cuda.empty_cache()

        h = self.rms_norm(h, self.get_weight(self.NORM), cfg['rms_eps'])
        lm_w = self.get_weight(self.LM_HEAD)
        if lm_w is None:
            lm_w = self.get_weight(self.EMBED)
        logits = h @ lm_w.t()
        if not self.cache_dequant:
            del lm_w; torch.cuda.empty_cache()
        return logits

    # ── Tokenizer ──

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
        log.info(f'Tokenizer loaded: vocab={self.tokenizer.vocab_size}')

    def tokenize(self, text):
        return torch.tensor([self.tokenizer.encode(text, add_special_tokens=False)],
                            dtype=torch.long, device=self.device)

    def get_logit_gap(self, prompt, correct, wrong, correct_id=None, wrong_id=None):
        logits = self.forward(self.tokenize(prompt))
        last = logits[0, -1, :]
        if correct_id is None:
            correct_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        if wrong_id is None:
            wrong_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        return last[correct_id].item() - last[wrong_id].item()

    def generate(self, prompt, max_tokens=15):
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(ids)
        for _ in range(max_tokens):
            out = self.forward(torch.tensor([generated], dtype=torch.long, device=self.device))
            generated.append(out[0, -1, :].argmax().item())
        return self.tokenizer.decode(generated)

    # ── Bit manipulation ──

    def flip_group(self, layer, proj_suffix, group_idx):
        name = self.layer_name(layer, proj_suffix)
        if name not in self._unpacked_cache:
            w = self.weights[name]
            out_f, in_f = w['shape'][1], w['shape'][0]
            packed, scales = self.unpack_q1_0(w['raw'], out_f, in_f)
            self._unpacked_cache[name] = (packed, scales, out_f, in_f)
        self._unpacked_cache[name][0][group_idx] ^= -1
        if name in self._dequant_cache:
            del self._dequant_cache[name]

    def flip_scale_bit(self, layer, proj_suffix, group_idx, bit_position):
        """Flip one mantissa bit in the FP16 scale of a weight group.
        bit_position: 4-9 (mantissa bits producing 1.5%-50% magnitude change).
        """
        name = self.layer_name(layer, proj_suffix)
        if name not in self._unpacked_cache:
            w = self.weights[name]
            out_f, in_f = w['shape'][1], w['shape'][0]
            packed, scales = self.unpack_q1_0(w['raw'], out_f, in_f)
            self._unpacked_cache[name] = (packed, scales, out_f, in_f)
        scales = self._unpacked_cache[name][1]
        scale_bits = scales.view(torch.int16)
        scale_bits[group_idx] ^= (1 << bit_position)
        if name in self._dequant_cache:
            del self._dequant_cache[name]

    def reset(self):
        self._unpacked_cache.clear()
        self._dequant_cache.clear()

# ─────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────

DEFAULT_EDITING_PROBES = [
    {'prompt': 'The spec says "permanently removes". The rule says soft-delete. Corrected: "',
     'correct': 'Sets', 'wrong': 'Perm', 'name': 'edit_softdelete'},
    {'prompt': 'Fix: "Returns all registered users" needs admin. Corrected: "Returns',
     'correct': ' the', 'wrong': ' a', 'name': 'edit_admin'},
    {'prompt': 'Remove the signing key from the response. Instead return: {"',
     'correct': 'token', 'wrong': 'server', 'name': 'edit_secret'},
    {'prompt': 'Add write scope: "Requires scope',
     'correct': ':', 'wrong': '.', 'name': 'edit_scope'},
    {'prompt': 'Bulk import needs admin. Change to: "Requires admin role',
     'correct': ' and', 'wrong': '.', 'name': 'edit_bulk'},
]

DEFAULT_CONTROL_PROBES = [
    {'prompt': 'The capital of France is', 'correct': ' Paris', 'wrong': ' London', 'name': 'geo_france'},
    {'prompt': '2 + 2 =', 'correct': ' 4', 'wrong': ' 5', 'name': 'math_basic'},
    {'prompt': 'The color of the sky is', 'correct': ' blue', 'wrong': ' red', 'name': 'knowledge'},
]


def load_probes(path):
    """Load probe file. Returns (target_probes, control_probes)."""
    with open(path) as f:
        data = json.load(f)
    targets = data['probes']['target']
    controls = data['probes'].get('promoted_controls', [])
    if not controls:
        controls = DEFAULT_CONTROL_PROBES
    log.info(f'Loaded {len(targets)} target + {len(controls)} control probes from {path}')
    return targets, controls

# ─────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────

def _score_probe(engine, p):
    """Logit-gap scorer that respects pre-calibrated token ids if present."""
    return engine.get_logit_gap(
        p['prompt'], p['correct'], p['wrong'],
        correct_id=p.get('base_correct_id'),
        wrong_id=p.get('base_wrong_id'),
    )


def run_search(engine, target_probes, control_probes, iterations=300, lambda_ctrl=2.0,
               auto_term_window=0, auto_term_rate=0.01, log_detail=False,
               scale_bits=None, scale_ratio=0.0):
    """Run XOR group search. Returns (flips, history).

    auto_term_window: if >0, stop when the acceptance rate over the last
        `auto_term_window` iterations drops below `auto_term_rate`.
    log_detail: if True, record per-probe gaps before/after in each accept.
    scale_bits: list of mantissa bit positions (4-9) to search. None = weight-only.
    scale_ratio: fraction of iterations that try scale flips (0.0-1.0).
    """
    scale_mode = f', scale_bits={scale_bits} ratio={scale_ratio}' if scale_bits else ''
    log.info(f'Search: {iterations} iters, {len(target_probes)}t + {len(control_probes)}c probes, '
             f'lambda={lambda_ctrl}, auto_term={auto_term_window}@{auto_term_rate}{scale_mode}')

    log.info('Measuring baselines...')
    t_bl = [_score_probe(engine, p) for p in target_probes]
    c_bl = [_score_probe(engine, p) for p in control_probes]

    for p, g in zip(target_probes, t_bl):
        log.info(f'  T {p["name"]:<20} {g:>+7.3f} {"CORRECT" if g > 0 else "WRONG"}')
    for p, g in zip(control_probes, c_bl):
        log.info(f'  C {p["name"]:<20} {g:>+7.3f} {"CORRECT" if g > 0 else "WRONG"}')

    prev_tg, prev_cg = list(t_bl), list(c_bl)

    def fitness():
        tg = [_score_probe(engine, p) for p in target_probes]
        cg = [_score_probe(engine, p) for p in control_probes]
        t_imp = sum(g - b for g, b in zip(tg, t_bl)) / len(tg)
        c_deg = sum(max(0, b - g) for g, b in zip(cg, c_bl)) / len(cg)
        return t_imp - lambda_ctrl * c_deg, tg, cg

    best_f = 0.0
    accepted = []
    hist = []
    accept_flags = []
    t0 = time.time()
    actual_iters = 0

    for i in range(iterations):
        actual_iters = i + 1
        layer = random.choice(engine.search_layers)
        proj = random.choice(engine.PROJ_LIST)
        name = engine.layer_name(layer, proj)

        if name not in engine._unpacked_cache:
            w = engine.weights[name]
            out_f, in_f = w['shape'][1], w['shape'][0]
            packed, scales = engine.unpack_q1_0(w['raw'], out_f, in_f)
            engine._unpacked_cache[name] = (packed, scales, out_f, in_f)

        packed, scales, _, _ = engine._unpacked_cache[name]
        probs = scales.float().abs()
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1).item()

        # Choose move type: weight XOR or scale-bit flip
        if scale_bits and random.random() < scale_ratio:
            bit = random.choice(scale_bits)
            move_type = 'scale'
            engine.flip_scale_bit(layer, proj, idx, bit)
        else:
            bit = None
            move_type = 'group'
            engine.flip_group(layer, proj, idx)

        new_f, tg, cg = fitness()

        if new_f > best_f:
            flip_entry = {'layer': layer, 'proj': proj, 'group': idx, 'type': move_type,
                          'iteration': i, 'fitness_before': best_f, 'fitness_after': new_f}
            if move_type == 'scale':
                flip_entry['bit'] = bit
            if log_detail:
                flip_entry['target_gaps_before'] = list(zip([p['name'] for p in target_probes], prev_tg))
                flip_entry['target_gaps_after'] = list(zip([p['name'] for p in target_probes], tg))
                flip_entry['control_gaps_before'] = list(zip([p['name'] for p in control_probes], prev_cg))
                flip_entry['control_gaps_after'] = list(zip([p['name'] for p in control_probes], cg))
            best_f = new_f
            accepted.append(flip_entry)
            prev_tg, prev_cg = list(tg), list(cg)
            el = time.time() - t0
            tag = f's{bit}' if move_type == 'scale' else 'w '
            log.info(f'  [{i:>3}] ACCEPT L{layer} {proj.split(".")[-1]:>10} g{idx:<6} {tag} f={new_f:>+.4f} ({el:.0f}s)')
            accept_flags.append(True)
        else:
            if move_type == 'scale':
                engine.flip_scale_bit(layer, proj, idx, bit)
            else:
                engine.flip_group(layer, proj, idx)
            accept_flags.append(False)

        hist.append(best_f)
        if (i + 1) % 25 == 0:
            log.info(f'  [{i+1:>3}] ... f={best_f:>+.4f}, accepted={len(accepted)}')

        # Auto-termination
        if auto_term_window > 0 and i >= auto_term_window:
            recent = sum(accept_flags[-auto_term_window:])
            if recent / auto_term_window < auto_term_rate:
                log.info(f'  [{i+1:>3}] AUTO-TERMINATE: {recent}/{auto_term_window} accepts '
                         f'({100*recent/auto_term_window:.1f}%) < {100*auto_term_rate:.0f}%')
                break

    elapsed = time.time() - t0
    rate = (elapsed / actual_iters) if actual_iters > 0 else 0.0
    log.info(f'Done: {len(accepted)} flips, fitness 0.0000 -> {best_f:.4f}, '
             f'{elapsed:.0f}s ({rate:.1f}s/iter, {actual_iters}/{iterations} iters)')

    log.info('Final probes:')
    for p, bl in zip(target_probes, t_bl):
        g = _score_probe(engine, p)
        tag = 'FIXED!' if bl <= 0 and g > 0 else ('BROKE!' if bl > 0 and g <= 0 else ('CORRECT' if g > 0 else 'WRONG'))
        log.info(f'  {p["name"]:<20} {bl:>+7.3f} -> {g:>+7.3f}  {tag}')
    for p, bl in zip(control_probes, c_bl):
        g = _score_probe(engine, p)
        log.info(f'  {p["name"]:<20} {bl:>+7.3f} -> {g:>+7.3f}  {"OK" if g > 0 else "DEGRADED"}')

    return accepted, hist


def export_patch(flips, history, iterations, search_layers, output_path, domain='editing'):
    """Write patch JSON."""
    patch = {
        'version': 2,
        'format': 'bankai_group_xor_v2',
        'base_model': 'prism-ml/Bonsai-8B-gguf',
        'domain': domain,
        'method': 'group-level greedy search',
        'search_layers': search_layers,
        'iterations': iterations,
        'flips': flips,
        'total_flips': len(flips),
        'bits_per_flip': 128,
        'total_bits_flipped': len(flips) * 128,
        'fitness_history': [float(h) for h in history],
    }
    with open(output_path, 'w') as f:
        json.dump(patch, f, indent=2)
    log.info(f'Exported: {output_path} ({len(flips)} flips, {os.path.getsize(output_path)} bytes)')

# ─────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────

def verify_generation(engine):
    """Quick coherence check."""
    log.info('Generation check:')
    for prompt in ['The capital of France is', '2 + 2 =']:
        text = engine.generate(prompt, max_tokens=15)
        log.info(f'  "{text}"')

# ─────────────────────────────────────────────────────────
# Pipeline (iterative multi-domain)
# ─────────────────────────────────────────────────────────

# Per-domain defaults. `iterations` overrides the CLI --iterations value
# when present. Rationale: calibrated failure rates differ widely across
# domains (see Plans/probe-generation-index.md Section 12.3), so instruction
# following deserves more search budget than math on Bonsai 8B.
PIPELINE_ORDER = [
    {'domain': 'math',         'probe_file': 'probes_math_math500.json'},
    {'domain': 'instruction',  'probe_file': 'probes_instruction.json'},
    {'domain': 'coding',       'probe_file': 'probes_coding.json'},
    {'domain': 'editing',      'probe_file': 'probes_editing.json'},
    {'domain': 'tool_calling', 'probe_file': 'probes_tool_calling.json'},
]

DOMAIN_PROBE_FILES = {s['domain']: s['probe_file'] for s in PIPELINE_ORDER}


MAX_ACCUMULATED_CONTROLS = 30  # cap on all_controls across pipeline steps


def _load_control_file(path):
    """Load a probe file and return all probes as a flat list (target + controls)."""
    with open(path) as f:
        data = json.load(f)
    probes = data['probes'].get('target', []) + data['probes'].get('promoted_controls', [])
    log.info(f'  Loaded {len(probes)} control probes from {path}')
    return probes


def run_pipeline(engine, iterations, output_dir, probes_dir='.',
                 per_domain_iters=None, per_domain_lambda=None,
                 max_controls=MAX_ACCUMULATED_CONTROLS,
                 pipeline_order=None, extra_control_files=None,
                 auto_term_window=0, auto_term_rate=0.01,
                 log_detail=False,
                 scale_bits=None, scale_ratio=0.0):
    """Run iterative multi-domain patching.

    `per_domain_iters`: dict mapping domain -> iteration count.
    `per_domain_lambda`: dict mapping domain -> lambda_ctrl value.
    `pipeline_order`: list of domain names to override PIPELINE_ORDER.
    `extra_control_files`: list of paths to probe JSON files whose probes
        are added to every domain's control set.
    `auto_term_window`/`auto_term_rate`: passed to run_search for early stop.
    `log_detail`: if True, record per-probe gaps in each accepted flip.
    `scale_bits`: list of mantissa bit positions (4-9) for scale search.
    `scale_ratio`: fraction of iterations that try scale flips.
    """
    os.makedirs(output_dir, exist_ok=True)
    per_domain_iters = per_domain_iters or {}
    per_domain_lambda = per_domain_lambda or {}

    # Build the ordered list of domains to run
    if pipeline_order:
        order = []
        for d in pipeline_order:
            pf = DOMAIN_PROBE_FILES.get(d)
            if pf:
                order.append({'domain': d, 'probe_file': pf})
            else:
                log.warning(f'Unknown domain in pipeline order: {d}')
    else:
        order = list(PIPELINE_ORDER)

    # Load extra control probes (MuSR, MMLU, etc.) that persist across domains
    base_controls = list(DEFAULT_CONTROL_PROBES)
    if extra_control_files:
        for cf in extra_control_files:
            if os.path.exists(cf):
                base_controls.extend(_load_control_file(cf))
    all_controls = list(base_controls)

    manifest = {'steps': [], 'search_layers': engine.search_layers,
                'probes_dir': probes_dir,
                'max_controls': max_controls,
                'pipeline_order': [s['domain'] for s in order]}

    for step in order:
        domain = step['domain']
        probe_file = os.path.join(probes_dir, step['probe_file'])

        if not os.path.exists(probe_file):
            log.warning(f'Skipping {domain}: {probe_file} not found')
            continue

        domain_iters = (per_domain_iters.get(domain)
                        or step.get('iterations')
                        or iterations)
        domain_lambda = per_domain_lambda.get(domain, 2.0)

        output_path = os.path.join(output_dir, f'patch_{domain}.json')

        log.info(f'\n{"="*60}')
        log.info(f'PIPELINE STEP: {domain.upper()} ({domain_iters} iters, lambda={domain_lambda})')
        log.info(f'{"="*60}')

        targets, promoted = load_probes(probe_file)
        controls = all_controls + promoted[:20]

        if os.path.exists(output_path):
            log.info(f'Resuming: {output_path} exists, replaying flips')
            with open(output_path) as f:
                existing = json.load(f)
            for flip in existing.get('flips', []):
                if flip.get('type') == 'scale':
                    engine.flip_scale_bit(flip['layer'], flip['proj'], flip['group'], flip['bit'])
                else:
                    engine.flip_group(flip['layer'], flip['proj'], flip['group'])
            flips = existing.get('flips', [])
            history = existing.get('fitness_history', [])
        else:
            flips, history = run_search(
                engine, targets, controls,
                iterations=domain_iters,
                lambda_ctrl=domain_lambda,
                auto_term_window=auto_term_window,
                auto_term_rate=auto_term_rate,
                log_detail=log_detail,
                scale_bits=scale_bits,
                scale_ratio=scale_ratio,
            )
            export_patch(flips, history, domain_iters, engine.search_layers, output_path, domain)

        # Promote passing targets to controls for next step
        for p in targets:
            g = _score_probe(engine, p)
            if g > 0:
                all_controls.append(p)
        if len(all_controls) > max_controls:
            keep = len(base_controls)
            tail = all_controls[keep:][-(max_controls - keep):]
            all_controls = list(base_controls) + tail
            log.info(f'Trimmed control set to {len(all_controls)} (cap={max_controls})')

        manifest['steps'].append({
            'domain': domain,
            'probe_file': probe_file,
            'patch_file': output_path,
            'flips': len(flips),
            'final_fitness': history[-1] if history else 0,
            'lambda': domain_lambda,
            'iterations_run': len(history),
            'controls_after': len(all_controls),
        })
        log.info(f'Controls for next step: {len(all_controls)}')

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log.info(f'\nPipeline complete. Manifest: {manifest_path}')

# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='XOR Patch Search for Bonsai 8B')
    parser.add_argument('--model', default='./Bonsai-8B.gguf', help='Path to GGUF model')
    parser.add_argument('--probes', default=None, help='Probe JSON file (default: built-in editing probes)')
    parser.add_argument('--output', default='patch.json', help='Output patch file')
    parser.add_argument('--iterations', type=int, default=300, help='Search iterations')
    parser.add_argument('--lambda-ctrl', type=float, default=2.0, help='Control penalty weight')
    parser.add_argument('--layers', type=int, nargs='+', default=DEFAULT_SEARCH_LAYERS, help='Search layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pipeline', action='store_true', help='Run iterative multi-domain pipeline')
    parser.add_argument('--pipeline-dir', default='./patches', help='Output dir for pipeline patches')
    parser.add_argument('--probes-dir', default='.', help='Directory containing probe JSON files for pipeline mode (e.g., ./calibrated)')
    parser.add_argument('--per-domain-iters', default=None,
                        help='Per-domain iteration overrides, e.g. "math=30,instruction=200". '
                             'Unspecified domains fall back to --iterations.')
    parser.add_argument('--per-domain-lambda', default=None,
                        help='Per-domain lambda overrides, e.g. "editing=1.5,instruction=1.0,coding=0.75". '
                             'Unspecified domains fall back to --lambda-ctrl.')
    parser.add_argument('--pipeline-order', default=None,
                        help='Comma-separated domain order, e.g. "editing,instruction,tool_calling,math,coding". '
                             'Overrides the built-in PIPELINE_ORDER.')
    parser.add_argument('--extra-control-files', nargs='*', default=None,
                        help='Additional probe JSON files whose probes are added to every domain\'s control set. '
                             'Used for MuSR/MMLU controls.')
    parser.add_argument('--max-controls', type=int, default=MAX_ACCUMULATED_CONTROLS,
                        help='Cap on accumulated controls across pipeline steps.')
    parser.add_argument('--auto-terminate-window', type=int, default=0,
                        help='If >0, stop search when acceptance rate over last N iters drops below --auto-terminate-rate.')
    parser.add_argument('--auto-terminate-rate', type=float, default=0.01,
                        help='Minimum acceptance rate threshold for auto-termination (default 0.01 = 1%%).')
    parser.add_argument('--log-flip-detail', action='store_true',
                        help='Record per-probe gap snapshots in each accepted flip (large output).')
    parser.add_argument('--scale-bits', type=str, default=None,
                        help='Comma-separated mantissa bit positions for scale search (e.g. "4,5,6,7,8,9")')
    parser.add_argument('--scale-ratio', type=float, default=0.7,
                        help='Fraction of iterations that try scale flips vs weight XOR (default 0.7)')
    parser.add_argument('--pre-patches', type=str, default=None,
                        help='Directory of patch files to apply before search (e.g. v2 patches for v3 run)')
    parser.add_argument('--verify', action='store_true', help='Run generation check before search')
    parser.add_argument('--no-download', action='store_true', help='Skip auto-download')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda')
    log.info(f'GPU: {torch.cuda.get_device_name()}')
    log.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    if not args.no_download:
        args.model = ensure_model(args.model)

    cfg, weights = load_model(args.model, device)
    engine = BonsaiEngine(cfg, weights, device, search_layers=args.layers)
    engine.load_tokenizer()

    if args.pre_patches:
        import glob
        patch_files = sorted(glob.glob(os.path.join(args.pre_patches, 'patch_*.json')))
        total_pre = 0
        for pf in patch_files:
            with open(pf) as f:
                patch = json.load(f)
            for flip in patch.get('flips', []):
                if flip.get('type') == 'scale':
                    engine.flip_scale_bit(flip['layer'], flip['proj'], flip['group'], flip['bit'])
                else:
                    engine.flip_group(flip['layer'], flip['proj'], flip['group'])
            total_pre += len(patch.get('flips', []))
            log.info(f'Pre-applied: {pf} ({len(patch.get("flips", []))} flips)')
        log.info(f'Total pre-applied: {total_pre} flips from {len(patch_files)} patches')

    if args.verify:
        verify_generation(engine)

    if args.pipeline:
        def _parse_kv(s, cast=int):
            d = {}
            if s:
                for kv in s.split(','):
                    if '=' not in kv:
                        continue
                    k, v = kv.split('=', 1)
                    d[k.strip()] = cast(v.strip())
            return d

        per_domain_iters = _parse_kv(args.per_domain_iters, int)
        per_domain_lambda = _parse_kv(args.per_domain_lambda, float)
        pipeline_order = [d.strip() for d in args.pipeline_order.split(',')] if args.pipeline_order else None

        run_pipeline(engine, args.iterations, args.pipeline_dir,
                     probes_dir=args.probes_dir,
                     per_domain_iters=per_domain_iters,
                     per_domain_lambda=per_domain_lambda,
                     max_controls=args.max_controls,
                     pipeline_order=pipeline_order,
                     extra_control_files=args.extra_control_files,
                     auto_term_window=args.auto_terminate_window,
                     auto_term_rate=args.auto_terminate_rate,
                     log_detail=args.log_flip_detail,
                     scale_bits=[int(b) for b in args.scale_bits.split(',')] if args.scale_bits else None,
                     scale_ratio=args.scale_ratio)
    else:
        if args.probes:
            targets, controls = load_probes(args.probes)
        else:
            targets = DEFAULT_EDITING_PROBES
            controls = DEFAULT_CONTROL_PROBES

        if args.extra_control_files:
            for cf in args.extra_control_files:
                if os.path.exists(cf):
                    controls = list(controls) + _load_control_file(cf)
                    log.info(f'Loaded extra controls from {cf} (total controls={len(controls)})')
                else:
                    log.warning(f'extra control file missing: {cf}')

        # Do NOT engine.reset() here: reset() clears _unpacked_cache, which is
        # exactly where --pre-patches wrote its flips. A reset here silently
        # undoes every pre-applied flip before run_search measures baselines,
        # making Task C sequential identical to independent per-domain runs.
        scale_bits = [int(b) for b in args.scale_bits.split(',')] if args.scale_bits else None
        flips, history = run_search(
            engine, targets, controls,
            iterations=args.iterations,
            lambda_ctrl=args.lambda_ctrl,
            auto_term_window=args.auto_terminate_window,
            auto_term_rate=args.auto_terminate_rate,
            log_detail=args.log_flip_detail,
            scale_bits=scale_bits,
            scale_ratio=args.scale_ratio,
        )
        export_patch(flips, history, args.iterations, args.layers, args.output)
        verify_generation(engine)


if __name__ == '__main__':
    main()
