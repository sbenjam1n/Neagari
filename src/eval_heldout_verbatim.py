#!/usr/bin/env python3
"""Held-out verbatim-extraction evaluation for Neagari v4.4.

Addresses referee Issue 27: the focused-fitness generation evaluation
in §6.8 is on the optimized probes themselves. This script runs the
same base/patched comparison on 20 held-out probes (same structural
template, different content) to test whether the patch installs a
generalizable 'extract rather than copy' capability or memorizes two
specific input->output mappings.

Usage:
    python3 scripts/eval_heldout_verbatim.py \\
        --model Bonsai-1.7B.gguf \\
        --patch overnight_results/focused_30k/patch_minimal.json \\
        --probes probes_verbatim_heldout.json \\
        --probes-training probes_verbatim_training.json \\
        --output-dir results/heldout_verbatim_eval \\
        --max-tokens 80

A --smoke-only flag runs only the 5 training probes and verifies
reproduction of generation_comparison.json::after. Use this before
running the full held-out eval.

Design: see Plans/v4_4-empirical-plan.md §2.

Dependencies: same as xor_search.py (gguf, transformers, torch, etc).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xor_search import BonsaiEngine, ensure_model, load_model

log = logging.getLogger('heldout_eval')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)


# ── Training probes (for smoke test) ─────────────────────────
# Verbatim copy of the VERBATIM_PROBES constant in overnight_verbatim_v2.py
# so that a smoke test does not require importing that module (which loads
# additional dependencies). Any drift between this list and the canonical
# list will be caught by the smoke test's reproduction check.
TRAINING_PROBES = [
    {
        'name': 'verbatim_legal_clause',
        'system': 'Extract the exact sentence from the passage that describes the liability limitation. Output ONLY the verbatim text, nothing else.',
        'prompt': 'Passage: "The agreement shall be governed by the laws of Delaware. Neither party shall be liable for consequential damages arising from performance under this agreement. All disputes shall be resolved through binding arbitration."\n\nExtract the sentence about liability limitation.',
        'target_span': 'Neither party shall be liable for consequential damages arising from performance under this agreement.',
        'passage': 'The agreement shall be governed by the laws of Delaware. Neither party shall be liable for consequential damages arising from performance under this agreement. All disputes shall be resolved through binding arbitration.',
    },
    {
        'name': 'verbatim_api_auth',
        'system': 'Extract the exact sentence from the documentation that describes the authentication method. Output ONLY the verbatim text, nothing else.',
        'prompt': 'Documentation: "Requests are rate-limited to 100 per minute. Authentication requires a Bearer token in the Authorization header with scope read:data. Response payloads are gzip-compressed by default."\n\nExtract the sentence about authentication.',
        'target_span': 'Authentication requires a Bearer token in the Authorization header with scope read:data.',
        'passage': 'Requests are rate-limited to 100 per minute. Authentication requires a Bearer token in the Authorization header with scope read:data. Response payloads are gzip-compressed by default.',
    },
    {
        'name': 'verbatim_error_msg',
        'system': 'Extract the exact error message from the log. Output ONLY the verbatim text, nothing else.',
        'prompt': 'Log output:\n[2026-04-14 03:22:15] INFO: Connection pool initialized with 50 workers\n[2026-04-14 03:22:18] ERROR: Failed to acquire lock on resource /db/users/schema — timeout after 30000ms\n[2026-04-14 03:22:18] INFO: Retrying with exponential backoff\n\nExtract the error message.',
        'target_span': 'Failed to acquire lock on resource /db/users/schema — timeout after 30000ms',
        'passage': '[2026-04-14 03:22:15] INFO: Connection pool initialized with 50 workers\n[2026-04-14 03:22:18] ERROR: Failed to acquire lock on resource /db/users/schema — timeout after 30000ms\n[2026-04-14 03:22:18] INFO: Retrying with exponential backoff',
    },
    {
        'name': 'verbatim_dosage',
        'system': 'Extract the exact sentence that states the recommended dosage. Output ONLY the verbatim text, nothing else.',
        'prompt': 'Prescribing information: "Administer orally once daily with food. The recommended starting dose is 10mg for adults over 18 years of age. Dose adjustments should be made at intervals of no less than 2 weeks."\n\nExtract the dosage sentence.',
        'target_span': 'The recommended starting dose is 10mg for adults over 18 years of age.',
        'passage': 'Administer orally once daily with food. The recommended starting dose is 10mg for adults over 18 years of age. Dose adjustments should be made at intervals of no less than 2 weeks.',
    },
    {
        'name': 'verbatim_function_sig',
        'system': 'Extract the exact function signature from the code. Output ONLY the verbatim text, nothing else.',
        'prompt': 'Source code:\n```python\nimport numpy as np\n\ndef compute_rolling_average(values: list[float], window_size: int = 7) -> list[float]:\n    """Compute a rolling average over the given window."""\n    return [np.mean(values[max(0,i-window_size+1):i+1]) for i in range(len(values))]\n```\n\nExtract the function signature (the def line).',
        'target_span': 'def compute_rolling_average(values: list[float], window_size: int = 7) -> list[float]:',
        'passage': 'import numpy as np\n\ndef compute_rolling_average(values: list[float], window_size: int = 7) -> list[float]:\n    """Compute a rolling average over the given window."""\n    return [np.mean(values[max(0,i-window_size+1):i+1]) for i in range(len(values))]',
    },
]

# Expected reproduction of generation_comparison.json::after for the 5 training
# probes under the 3,019-flip focused patch. If the smoke test's patched
# outputs don't match this, patch application has drifted.
EXPECTED_PATCHED = {
    'verbatim_legal_clause': 'Neither party shall be liable for consequential damages arising from performance under this agreement.',
    'verbatim_api_auth': 'Authentication requires a Bearer token in the Authorization header with scope read:data.',
    'verbatim_error_msg': 'Failed to acquire lock on resource /db/users/schema — timeout after 30000ms',
    'verbatim_dosage': 'The recommended starting dose is 10mg for adults over 18 years of age.',
    'verbatim_function_sig': 'def compute_rolling_average(values: list[float], window_size: int = 7) -> list[float]:',
}


def build_prompt(probe, tokenizer):
    """Format probe into a chat-template prompt, matching overnight_verbatim_v2.py.

    Uses the tokenizer's apply_chat_template with enable_thinking=False
    to suppress Qwen3's <think> reasoning mode, which would corrupt the
    generation output with reasoning tokens.
    """
    messages = [
        {'role': 'system', 'content': probe['system']},
        {'role': 'user', 'content': probe['prompt']},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        # Older transformers without enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def classify(output_text, target_span, passage):
    """Score a single output into PASS / COPY / PARTIAL / OTHER.

    PASS: output contains target_span AND does NOT contain passage content
          beyond the target_span (i.e., the model extracted correctly).
    COPY: output contains passage content beyond the target_span (the model
          dumped the passage instead of extracting).
    PARTIAL: some fragment of the passage appears but neither PASS nor COPY.
    OTHER: unrelated output.

    The "beyond target_span" check avoids a false COPY classification when
    the target sentence happens to be the first sentence of the passage
    (e.g., heldout_legal_warranty).
    """
    out = output_text.strip()
    passage_clean = passage.strip()
    has_target = target_span in out

    # COPY check: does the output contain passage content BEYOND the target?
    # Remove the target_span from the passage to isolate "extra" text, then
    # check whether any 15-char fragment of that extra text appears in output.
    extra = passage_clean.replace(target_span, '').strip()
    has_extra = False
    if len(extra) >= 15:
        for i in range(0, len(extra) - 14):
            if extra[i:i + 15] in out:
                has_extra = True
                break

    if has_target and not has_extra:
        return 'PASS'
    if has_extra:
        return 'COPY'

    # Check partial: any 15-char fragment of the full passage in the output
    if len(passage_clean) >= 15:
        for i in range(0, len(passage_clean) - 14):
            if passage_clean[i:i + 15] in out:
                return 'PARTIAL'
    return 'OTHER'


def generate_on_probes(engine, probes, max_tokens):
    import re
    results = []
    for probe in probes:
        prompt_text = build_prompt(probe, engine.tokenizer)
        t0 = time.time()
        output = engine.generate(prompt_text, max_tokens=max_tokens)
        dt = time.time() - t0
        # Strip the prompt from the output; engine.generate decodes the full sequence
        if output.startswith(prompt_text):
            gen_only = output[len(prompt_text):]
        else:
            # tokenization round-trip may add spaces; fall back to assistant-tag split
            if '<|im_start|>assistant\n' in output:
                gen_only = output.split('<|im_start|>assistant\n', 1)[1]
            else:
                gen_only = output
        # Trim at end-of-turn if present
        for stop in ('<|im_end|>', '<|endoftext|>'):
            if stop in gen_only:
                gen_only = gen_only.split(stop, 1)[0]
        # Strip <think>...</think> blocks (Qwen3 reasoning mode safety net)
        gen_only = re.sub(r'<think>.*?</think>', '', gen_only, flags=re.DOTALL).strip()
        category = classify(gen_only, probe['target_span'], probe['passage'])
        results.append({
            'name': probe['name'],
            'output': gen_only.strip(),
            'category': category,
            'wall_s': round(dt, 3),
        })
        log.info(f'  {probe["name"]:35s}  [{category}]  {gen_only.strip()[:80]!r}')
    return results


def apply_patch(engine, patch_flips):
    log.info(f'Applying {len(patch_flips)} sign-group flips...')
    t0 = time.time()
    for layer, proj, group in patch_flips:
        engine.flip_group(layer, proj, group)
    log.info(f'Patch applied in {time.time() - t0:.1f}s')


def reset_engine(engine):
    engine.reset()


def cross_tab(base_results, patched_results):
    """Return count table over (base, patched) categories."""
    cats = ['PASS', 'COPY', 'PARTIAL', 'OTHER']
    tab = {b: {p: 0 for p in cats} for b in cats}
    for base_r, patched_r in zip(base_results, patched_results):
        tab[base_r['category']][patched_r['category']] += 1
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Bonsai-1.7B.gguf')
    ap.add_argument('--patch', required=True, help='Path to patch_minimal.json')
    ap.add_argument('--probes', help='Held-out probes JSON file')
    ap.add_argument('--output-dir', default='results/heldout_verbatim_eval')
    ap.add_argument('--max-tokens', type=int, default=80)
    ap.add_argument('--smoke-only', action='store_true',
                    help='Run only the 5 training probes and verify reproduction; do not run held-out')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f'=== Neagari held-out verbatim eval ===')
    log.info(f'Model: {args.model}')
    log.info(f'Patch: {args.patch}')

    # Load model
    model_path = ensure_model(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')
    cfg, weights = load_model(model_path, device)
    engine = BonsaiEngine(cfg, weights, device)
    engine.load_tokenizer()

    # Load patch
    patch = json.load(open(args.patch))
    flips = patch['flips']
    log.info(f'Patch has {len(flips)} flips')

    # Canary check before patching
    canary = engine.generate('The capital of France is', max_tokens=10)
    log.info(f'Canary (base): {canary!r}')

    # === SMOKE TEST: reproduce training-probe generation_comparison.json::after ===
    log.info('=== Smoke test: 5 training probes ===')
    log.info('Base inference on training probes:')
    base_training = generate_on_probes(engine, TRAINING_PROBES, args.max_tokens)
    apply_patch(engine, flips)
    log.info(f'Canary (patched): {engine.generate("The capital of France is", max_tokens=10)!r}')
    log.info('Patched inference on training probes:')
    patched_training = generate_on_probes(engine, TRAINING_PROBES, args.max_tokens)

    # Verify reproduction
    smoke_ok = True
    for pr in patched_training:
        expected = EXPECTED_PATCHED.get(pr['name'], '')
        if expected and expected not in pr['output']:
            log.error(f'SMOKE FAIL: {pr["name"]}')
            log.error(f'  expected: {expected!r}')
            log.error(f'  got:      {pr["output"]!r}')
            smoke_ok = False
    if smoke_ok:
        log.info('SMOKE PASS: all 5 training probes reproduce expected patched output.')
    else:
        log.error('SMOKE FAIL: patch application does not reproduce stored results. ABORT.')
        json.dump({
            'smoke_pass': False,
            'training': {
                'base': base_training, 'patched': patched_training,
            },
        }, open(out_dir / 'smoke.json', 'w'), indent=2)
        sys.exit(1)

    json.dump({
        'smoke_pass': True,
        'training': {'base': base_training, 'patched': patched_training},
    }, open(out_dir / 'smoke.json', 'w'), indent=2)

    if args.smoke_only:
        log.info('Smoke-only mode; exiting.')
        return

    # === HELD-OUT EVAL ===
    if not args.probes:
        log.error('--probes required unless --smoke-only')
        sys.exit(2)
    probeset = json.load(open(args.probes))
    probes = probeset['probes']
    log.info(f'=== Held-out eval: {len(probes)} probes ===')

    # Need to regenerate on held-out. Engine is currently patched; reset to base first.
    reset_engine(engine)
    log.info('Base inference on held-out probes:')
    base_heldout = generate_on_probes(engine, probes, args.max_tokens)

    # Re-apply patch
    apply_patch(engine, flips)
    log.info('Patched inference on held-out probes:')
    patched_heldout = generate_on_probes(engine, probes, args.max_tokens)

    # Score
    tab = cross_tab(base_heldout, patched_heldout)
    base_copy = sum(tab['COPY'].values())
    copy_to_pass = tab['COPY']['PASS']
    pass_to_copy = tab['PASS']['COPY']
    pass_to_pass = tab['PASS']['PASS']
    total_pass_base = sum(r['category'] == 'PASS' for r in base_heldout)
    total_pass_patched = sum(r['category'] == 'PASS' for r in patched_heldout)

    summary = {
        'n_probes': len(probes),
        'cross_tab': tab,
        'base_pass_count': total_pass_base,
        'patched_pass_count': total_pass_patched,
        'delta_pass': total_pass_patched - total_pass_base,
        'base_copy_count': base_copy,
        'copy_to_pass_count': copy_to_pass,
        'pass_to_copy_count': pass_to_copy,
        'pass_to_pass_count': pass_to_pass,
        'copy_to_pass_ratio': (copy_to_pass / base_copy) if base_copy else None,
        'interpretation': (
            'strong generalization' if base_copy and copy_to_pass / base_copy > 0.5 else
            'mild generalization' if base_copy and copy_to_pass / base_copy > 0.2 else
            'weak or no generalization' if base_copy else
            'no COPY cases on base (uninformative); consider harder probes'
        ),
    }

    log.info('=== Summary ===')
    log.info(f'Base PASS:    {total_pass_base}/{len(probes)}')
    log.info(f'Patched PASS: {total_pass_patched}/{len(probes)}')
    log.info(f'COPY on base that became PASS after patch: {copy_to_pass}/{base_copy} '
             f'({summary["copy_to_pass_ratio"]})')
    log.info(f'Interpretation: {summary["interpretation"]}')
    log.info(f'Cross-tab (rows=base, cols=patched):')
    for b_cat, row in tab.items():
        log.info(f'  {b_cat:8s}  {row}')

    # Dump artifacts
    with open(out_dir / 'generations.jsonl', 'w') as f:
        for state, rs in [('base', base_heldout), ('patched', patched_heldout)]:
            for r in rs:
                f.write(json.dumps({'state': state, **r}) + '\n')
    json.dump(summary, open(out_dir / 'summary.json', 'w'), indent=2)
    log.info(f'Artifacts written to {out_dir}/')


if __name__ == '__main__':
    main()
