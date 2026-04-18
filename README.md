# Neagari

**Navigable Degeneracy in the Roots of 1-Bit Language Models**

Gradient-free adaptation of frozen 1-bit language models via discrete search over binary weight groups.

[**Paper (PDF)**](paper/neagari-paper.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19630592.svg)](https://doi.org/10.5281/zenodo.19630592)

This Colab notebook demonstrates a 13 KB patch on Bonsai 1.7B that corrects two verbatim text extraction failures. You can run it for free on T4 after waiting for Bonsai to download. The patch was found using a novel targeted search that concentrates on probes nearest the decision boundary (focal-loss-style weighting).

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ozwjKPSCNewuoR9w9ziHs1oJUpy0d8U)

## What this work finds

We find that the binary weight space of true 1-bit language models (one sign bit per weight, shared FP16 scale per group) contains a structural property we call navigable degeneracy: 27–47% of random sign-group perturbations in MLP layers improve task-specific logit gaps while preserving general performance, validated against a null baseline on randomized weights (46.8% vs 16.8% acceptance, 30pp gap with non-overlapping CIs).

The central finding is a fitness-behavior gap that operates at two scales. At the probe level, 99.96% of accepted flips under an average-gap fitness function produce no change in any probe's argmax prediction, with per-flip effect sizes four orders of magnitude below typical decision margins. At the benchmark level, we do not detect a statistically significant effect on any of the four benchmarks we evaluated (GSM8K shows a directional signal at p=0.110 with a confidence interval that includes zero; the other three are flat). The landscape is navigable by the fitness metric but the navigation does not produce detectable behavioral change under uniform fitness weighting.

We trace this to fitness dilution: the average-gap criterion distributes credit uniformly across probes, so the search drifts laterally across a neutral network in the Kimura (1968) sense without accumulating directional progress toward any specific decision boundary. A boundary-concentrated fitness function, applying inverse-margin weighting inspired by focal loss to discrete binary search, resolves this at the probe level by creating a selection gradient toward near-boundary probes. The focused variant crosses both targeted probes by iteration 6,059 on Bonsai 1.7B. A held-out evaluation on 100 same-structure probes finds 8% conversion (95% CI [4%, 16%]), below the pre-registered 20% threshold, with all conversions concentrated in the two training-target domains. The result is consistent with memorization of the optimized mappings rather than installation of a transferable capability..

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

# With per-flip per-probe delta logging (for diffusion analysis):
python src/xor_search.py \
    --model Bonsai-8B-v1.gguf \
    --probes your_probes.json \
    --iterations 500 \
    --log-flip-detail
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
│   ├── xor_search.py                # Core search engine (LLM)
│   ├── neagari_vision.py            # Vision search engine (Binary ResNet-18 / CIFAR-10)
│   ├── apply_patches_gguf.py        # Patch application utility
│   ├── eval_heldout_verbatim.py     # Held-out verbatim evaluation
│   └── preflight_vision.py          # Pre-run validation for vision experiments
├── patches/
│   ├── v2_patches/                  # Weight-XOR patches (5 domains)
│   └── v3_patches/                  # Scale-mantissa patches (5 domains)
└── probes/
    ├── calibrated/                  # Calibrated probe sets (5 domains)
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
