# Neagari

**Navigable Degeneracy in the Roots of 1-Bit Language Models**

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ozwjKPSCNewuoR9w9ziHs1oJUpy0d8U)
&ensp;

[**Paper (PDF)**](paper/neagari-paper.pdf)

[![DOI](https://zenodo.org/badge/1209991674.svg)](https://doi.org/10.5281/zenodo.19615053)

Gradient-free adaptation of frozen 1-bit language models via discrete search over binary weight groups. 2,252 patches across 5 task domains on PrismML's Bonsai 8B, applied via bitwise XOR in microseconds, preserving the native 1-bit inference path. 17.2 A100 GPU-hours total compute.

## What this work finds

True 1-bit language models store each weight as a single sign bit with a shared scale factor per group. We find that the binary weight space of these models contains what we call navigable degeneracy: 27–47% of random sign-group perturbations in MLP layers improve task-specific logit gaps while preserving general performance. A null-baseline comparison confirms this is a property of the trained model's structure rather than an artifact of the fitness criterion (46.8% acceptance on the trained model versus 16.8% on randomized weights).

However, navigating this landscape is subtler than acceptance rates suggest. Of 2,252 accepted flips under an average-gap fitness function, 99.96% produce no change in any probe's argmax prediction: per-flip effect sizes are four orders of magnitude below typical decision margins. At the benchmark level, we do not detect a statistically significant effect on any of the four benchmarks we evaluated. The fitness function rewards movement that is too small to cross decision boundaries, a failure mode we call fitness dilution.

A boundary-concentrated fitness function, inspired by focal loss, resolves this at the probe level by concentrating search pressure on probes nearest the decision boundary. The focused variant crosses both targeted probes and produces observable generation changes on those two specific prompts under a 13 KB patch. A held-out evaluation (below) finds that these changes do not generalize beyond the training-target domains, consistent with memorization of the optimized mappings rather than installation of a transferable capability.

## Held-out verbatim evaluation (1.7B)

The demo notebook applies a focused patch trained on two verbatim extraction prompts (legal clause and medical dosage). To test whether the patch installs a general "extract rather than copy" capability or memorizes two specific input-output mappings, we evaluated on 100 held-out probes (20 per domain) that follow the same structural template but use different content.

|  | Patched PASS | Patched COPY | Patched PARTIAL |
|---|---|---|---|
| **Base PASS** | 19 | 1 | 0 |
| **Base COPY** | 6 | 72 | 0 |
| **Base PARTIAL** | 0 | 0 | 2 |

COPY→PASS conversion rate: 6/78 = 7.7% (Wilson 95% CI [3.6%, 15.8%]), below the pre-registered 20% threshold for generalization. Per-domain breakdown:

| Domain | Base COPY | Converted | Rate |
|--------|-----------|-----------|------|
| Legal | 15 | 3 | 20% |
| Medical | 16 | 3 | 19% |
| API | 14 | 0 | 0% |
| Code | 14 | 0 | 0% |
| Logs | 19 | 0 | 0% |

All six conversions are in the two domains that correspond to the two training targets (legal clause and medical dosage). Zero conversions in the other three domains. One base-PASS probe reverted to COPY under the patch (5% breakage on base-PASS probes). The result is consistent with memorization of the two optimized mappings with within-domain spillover rather than a task-general capability shift.

## Search results (8B)

Weight-XOR search across five task domains on Bonsai 8B. Each domain uses a different (λ, control set) configuration; per-domain rates are conditioned on these post-hoc choices and on the sequential pipeline order (editing → instruction → tool calling → math → coding). See the paper §6 preamble and §6.5 footnote for details.

| Domain | λ | Controls | Iters | Flips | Rate |
|--------|---|----------|-------|-------|------|
| Editing | 1.5 | 33 | 1,000 | 469 | 46.9% |
| Tool calling | 1.0 | 40 | 500 | 233 | 46.6% |
| Instruction | 1.0 | 35 | 1,000 | 414 | 41.4% |
| Coding | 0.75 | 40 | 200 | 68 | 34.0% |
| Math | 2.0 | 40 | 200 | 54 | 27.0% |

A null-baseline comparison on Bonsai 1.7B confirms these rates reflect trained-model structure rather than fitness-criterion symmetry: 46.8% acceptance on the trained model versus 16.8% on a randomized model (30pp gap, non-overlapping 95% CIs).

## Benchmark evaluation (8B)

Base vs. patched, within-harness comparison (same prompts, same server build, same scoring):

| Benchmark | Base | Patched | Δ | n |
|-----------|------|---------|---|---|
| IFEval (prompt strict) | 77.63% | 77.82% | +0.19pp | 541 |
| GSM8K | 85.29% | 86.05% | +0.76pp | 1,319 |
| MMLU-Redux | 55.63% | 55.60% | −0.03pp | 14,042 |
| MuSR | 55.2% | 55.2% | 0.00pp | 250 |

No benchmark shows a statistically significant effect. GSM8K shows a directional signal (paired McNemar p=0.110, 95% CI [−0.08, +1.60]pp) whose confidence interval includes zero. The other three are flat within sampling noise.

## Quickstart: Apply Patches to Your Bonsai 8B

```bash
git clone https://github.com/sbenjam1n/neagari.git
cd neagari

# Download Bonsai 8B GGUF from HuggingFace
# (requires huggingface-cli or manual download from PrismML/Bonsai-8B-v1-GGUF)

# Apply all patches (weight-XOR + scale-mantissa, all 5 domains)
python src/apply_patches_gguf.py \
    --model Bonsai-8B-v1.gguf \
    --patches patches/v2_patches patches/v3_patches \
    --output Bonsai-8B-v1-neagari.gguf

# The patched GGUF is a drop-in replacement for any Q1_0-compatible backend
```

## Run the Search on Your Own Probes

```bash
pip install gguf transformers torch numpy huggingface_hub

# Design your probes as JSON: {"probes": [{"prompt": "...", "correct": "tok", "wrong": "tok", "name": "..."}]}
# Then run:
python src/xor_search.py \
    --model Bonsai-8B-v1.gguf \
    --probes your_probes.json \
    --iterations 500 \
    --lambda 1.5
```

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| `neagari_demo_1_7b.ipynb` | End-to-end demo on Bonsai 1.7B: load model, apply patch, compare base vs. patched generation on verbatim extraction probes. Runs on free Colab T4. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ozwjKPSCNewuoR9w9ziHs1oJUpy0d8U) |
| `boundary_crossing_audit.ipynb` | Reproduces the §7.6 boundary crossing audit: analyzes all 2,252 accepted flips, computes per-flip effect sizes, and shows why 99.96% produce no behavioral change. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15fRjHsIbYRQcruGjjMyFWirEYv5SLTEE) |
| `bankai_verification_standalone.ipynb` | Independent verification of Bankai (Saravanan, 2026) using a self-contained PyTorch Q1_0 inference engine. No external dependencies beyond torch and gguf. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pixNNRZWg3aFwngCXxwGwgGnd26jDNmH) |

## Repository Structure

```
neagari/
├── README.md
├── LICENSE                          # MIT
├── citation.cff
├── paper/
│   └── neagari-paper.pdf            # Full paper (v5.0)
├── notebooks/
│   ├── neagari_demo_1_7b.ipynb      # 1.7B demo (free Colab T4)
│   ├── boundary_crossing_audit.ipynb
│   └── bankai_verification_standalone.ipynb
├── src/
│   ├── xor_search.py                # Core search engine
│   ├── apply_patches_gguf.py        # Patch application utility
│   └── eval_heldout_verbatim.py     # Held-out verbatim evaluation
├── patches/
│   ├── v2_patches/                  # Weight-XOR patches (5 domains)
│   └── v3_patches/                  # Scale-mantissa patches (5 domains)
└── probes/
    ├── calibrated/                  # Calibrated probe sets (4 domains)
    └── probes_verbatim_heldout.json # 100 held-out verbatim probes
```

## Citation

```bibtex
@software{benjamin2026neagari,
  author = {Benjamin, Steven},
  title = {Neagari: Navigable Degeneracy in the Roots of 1-Bit Language Models},
  year = {2026},
  url = {https://github.com/sbenjam1n/neagari}
}
```

## License

MIT
