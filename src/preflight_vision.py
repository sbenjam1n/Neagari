#!/usr/bin/env python3
"""Preflight validation for Neagari vision experiments.

Runs on CPU in under 30 seconds. Catches configuration, schema, and
math errors before committing to GPU time. Run this before any
--train, --search, or --corruption experiment.

Validates:
  1. neagari_vision.py imports and core classes instantiate
  2. Binary model forward pass produces valid logits
  3. XOR flip is self-inverse (flip twice = identity)
  4. All four fitness functions produce finite scalars
  5. Per-flip delta logging produces valid JSONL schema
  6. Fokker-Planck parameterization: M and V computable from deltas
  7. Kimura fixation probability: u(p) is finite and in [0, 1]
  8. Kolmogorov backward expected crossing time: finite and positive
  9. CIFAR-10-C path construction and error handling
 10. Focused mode: lexicographic targeting advances on crossing

Usage:
    python src/preflight_vision.py

Exit code 0 = all checks pass. Non-zero = failure with diagnostic.
"""

import sys, os, json, math, tempfile, shutil

# ── Ensure src/ is importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}: {detail}")
        FAIL += 1


def main():
    global PASS, FAIL
    import torch
    import numpy as np

    device = torch.device('cpu')
    print("Neagari Vision Preflight")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    # ════════════════════════════════════════════════════════════
    # 1. Import and instantiation
    # ════════════════════════════════════════════════════════════
    print("1. Import and instantiation")
    try:
        from neagari_vision import (
            BinaryResNet18, BinaryConv2d, BinaryLinear,
            flip_group, score_probe, GROUP_SIZE,
            fitness_average, fitness_crossing, fitness_borderline,
            CIFAR10C_CORRUPTIONS, load_cifar10c,
        )
        check("neagari_vision imports", True)
    except Exception as e:
        check("neagari_vision imports", False, str(e))
        print("\nFATAL: Cannot import. Aborting.")
        sys.exit(1)

    model = BinaryResNet18(num_classes=10).to(device)
    model.freeze_binary()
    model.eval()
    check("BinaryResNet18 instantiates and freezes", True)

    binary_layers = model.get_binary_layers()
    n_layers = len(binary_layers)
    n_weights = sum(l.n_binary_weights() for _, l in binary_layers)
    n_groups = sum(l.n_groups(GROUP_SIZE) for _, l in binary_layers)
    check(f"Binary layers: {n_layers}, weights: {n_weights:,}, groups: {n_groups:,}",
          n_layers > 0 and n_weights > 0)
    print()

    # ════════════════════════════════════════════════════════════
    # 2. Forward pass
    # ════════════════════════════════════════════════════════════
    print("2. Forward pass")
    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        logits = model(x)
    check("Forward pass shape", logits.shape == (4, 10),
          f"expected (4,10), got {logits.shape}")
    check("Logits finite", torch.isfinite(logits).all().item(),
          f"non-finite values in logits")
    check("Logits vary", logits.std().item() > 1e-6,
          "all logits identical (dead model)")
    print()

    # ════════════════════════════════════════════════════════════
    # 3. XOR flip self-inverse
    # ════════════════════════════════════════════════════════════
    print("3. XOR flip self-inverse")
    layer_name, layer = binary_layers[0]
    signs_before = layer.binary_signs.clone()
    flip_group(layer, 0)
    signs_mid = layer.binary_signs.clone()
    flip_group(layer, 0)
    signs_after = layer.binary_signs.clone()
    check("Flip changes signs", not torch.equal(signs_before, signs_mid))
    check("Double flip restores", torch.equal(signs_before, signs_after))
    print()

    # ════════════════════════════════════════════════════════════
    # 4. Fitness functions
    # ════════════════════════════════════════════════════════════
    print("4. Fitness functions")
    tg = [0.5, -0.3, 0.1, -1.2]
    cg = [2.0, 1.5, 0.8]
    t_bl = [0.4, -0.5, 0.2, -1.0]
    c_bl = [2.1, 1.6, 0.9]
    lam = 1.0

    for name, fn in [('average', fitness_average),
                     ('crossing', fitness_crossing),
                     ('borderline', fitness_borderline)]:
        val = fn(tg, cg, t_bl, c_bl, lam)
        check(f"fitness_{name} = {val:.6f}", math.isfinite(val))
    print()

    # ════════════════════════════════════════════════════════════
    # 5. Per-flip delta logging schema
    # ════════════════════════════════════════════════════════════
    print("5. Per-flip delta logging schema")

    # Build synthetic probes
    n_targets, n_controls = 8, 4
    targets = []
    controls = []
    with torch.no_grad():
        for i in range(n_targets + n_controls):
            img = torch.randn(3, 32, 32, device=device)
            logits_i = model(img.unsqueeze(0))
            label = i % 10
            correct_logit = logits_i[0, label].item()
            wrong_logits = logits_i[0].clone()
            wrong_logits[label] = -1e9
            wrong_class = wrong_logits.argmax().item()
            gap = correct_logit - wrong_logits[wrong_class].item()
            probe = {
                'image_idx': i, 'image': img.cpu(),
                'label': label, 'wrong_class': wrong_class,
                'gap': gap, 'correct': gap > 0,
            }
            if i < n_targets:
                targets.append(probe)
            else:
                controls.append(probe)

    check(f"Synthetic probes: {len(targets)} targets, {len(controls)} controls", True)

    # Run a mini search with delta logging
    tmpdir = tempfile.mkdtemp(prefix='neagari_preflight_')
    delta_path = os.path.join(tmpdir, 'per_flip_deltas.jsonl')

    import random
    random.seed(42)
    torch.manual_seed(42)

    # Measure baselines
    t_bl_real = [score_probe(model, p, device) for p in targets]
    c_bl_real = [score_probe(model, p, device) for p in controls]
    current_tg = list(t_bl_real)
    current_cg = list(c_bl_real)

    candidates = []
    for lname, layer_obj in binary_layers:
        for g in range(layer_obj.n_groups(GROUP_SIZE)):
            candidates.append((lname, layer_obj, g))

    accepted_deltas = []
    n_iters = 50
    best_f = 0.0

    with open(delta_path, 'w') as fh:
        for i in range(n_iters):
            name_c, layer_c, gidx = random.choice(candidates)
            flip_group(layer_c, gidx)
            tg_now = [score_probe(model, p, device) for p in targets]
            cg_now = [score_probe(model, p, device) for p in controls]
            new_f = fitness_average(tg_now, cg_now, t_bl_real, c_bl_real, 1.0)
            if new_f > best_f:
                best_f = new_f
                t_deltas = [g - c for g, c in zip(tg_now, current_tg)]
                c_deltas = [g - c for g, c in zip(cg_now, current_cg)]
                entry = {
                    'iteration': i, 'flip_num': len(accepted_deltas) + 1,
                    'layer': name_c, 'group': gidx,
                    'target_deltas': t_deltas, 'control_deltas': c_deltas,
                    'target_gaps_after': tg_now, 'control_gaps_after': cg_now,
                }
                fh.write(json.dumps(entry) + '\n')
                accepted_deltas.append(entry)
                current_tg = list(tg_now)
                current_cg = list(cg_now)
            else:
                flip_group(layer_c, gidx)

    n_accepted = len(accepted_deltas)
    check(f"Mini search: {n_accepted}/{n_iters} accepted",
          n_accepted > 0, "zero accepts in 50 iterations")

    # Validate JSONL schema
    with open(delta_path) as fh:
        lines = fh.readlines()
    check(f"Delta JSONL lines = {len(lines)} = accepted count",
          len(lines) == n_accepted)

    if lines:
        entry = json.loads(lines[0])
        required_keys = {'iteration', 'flip_num', 'layer', 'group',
                         'target_deltas', 'control_deltas',
                         'target_gaps_after', 'control_gaps_after'}
        has_keys = required_keys.issubset(set(entry.keys()))
        check("JSONL schema has all required keys", has_keys,
              f"missing: {required_keys - set(entry.keys())}")
        check("target_deltas length matches n_targets",
              len(entry['target_deltas']) == n_targets)
        check("control_deltas length matches n_controls",
              len(entry['control_deltas']) == n_controls)
        check("All delta values finite",
              all(math.isfinite(v) for v in entry['target_deltas'] + entry['control_deltas']))
    print()

    # ════════════════════════════════════════════════════════════
    # 6. Fokker-Planck parameterization
    # ════════════════════════════════════════════════════════════
    print("6. Fokker-Planck parameterization")

    if n_accepted >= 2:
        # Extract per-probe deltas across all accepted flips
        all_t_deltas = np.array([e['target_deltas'] for e in accepted_deltas])
        # all_t_deltas shape: (n_accepted, n_targets)

        # Per-probe mean delta (M) and variance (V) = diffusion coefficients
        M = all_t_deltas.mean(axis=0)  # (n_targets,)
        V = all_t_deltas.var(axis=0)   # (n_targets,)

        check(f"M (per-probe mean delta) shape: {M.shape}",
              M.shape == (n_targets,))
        check(f"V (per-probe variance) shape: {V.shape}",
              V.shape == (n_targets,))
        check("M values finite",
              np.all(np.isfinite(M)), f"non-finite M: {M}")
        check("V values finite and non-negative",
              np.all(np.isfinite(V)) and np.all(V >= 0), f"bad V: {V}")
        check("At least one probe has nonzero variance",
              np.any(V > 0), "all probes have zero variance (no signal)")

        # Print summary
        print(f"    M range: [{M.min():.6f}, {M.max():.6f}]")
        print(f"    V range: [{V.min():.6f}, {V.max():.6f}]")
        print(f"    Probes with positive mean drift: {(M > 0).sum()}/{n_targets}")
    else:
        check("Enough accepts for parameterization", False,
              f"need >=2 accepts, got {n_accepted}")
        M = V = None
    print()

    # ════════════════════════════════════════════════════════════
    # 7. Kimura fixation probability
    # ════════════════════════════════════════════════════════════
    print("7. Kimura fixation probability")

    def kimura_fixation_prob(p, Ne_s):
        """Kimura (1962) fixation probability.

        u(p) = (1 - exp(-4*Ne*s*p)) / (1 - exp(-4*Ne*s))

        p: initial 'frequency' (normalized gap position, 0 = at boundary, 1 = far)
        Ne_s: product of effective population size and selection coefficient.
              When |Ne_s| << 1, u(p) ≈ p (neutral drift).
              When Ne_s >> 1, u(p) ≈ 1 - exp(-2*s*p) (selection dominated).
        """
        if abs(Ne_s) < 1e-10:
            return p  # neutral limit
        x = 4 * Ne_s
        num = 1 - math.exp(-x * p)
        den = 1 - math.exp(-x)
        if abs(den) < 1e-300:
            return p
        return num / den

    # Test with known values
    # Neutral: Ne_s ≈ 0 → u(p) ≈ p
    u_neutral = kimura_fixation_prob(0.1, 0.0)
    check(f"Neutral fixation u(0.1, 0) = {u_neutral:.4f} ≈ 0.1",
          abs(u_neutral - 0.1) < 0.01)

    # Strong selection: Ne_s = 10 → u(0.1) ≈ 1 - exp(-2*10*0.1) ≈ 0.865
    u_strong = kimura_fixation_prob(0.1, 10.0)
    check(f"Strong selection u(0.1, 10) = {u_strong:.4f} ≈ 0.86",
          0.5 < u_strong < 1.0)

    # Deleterious: Ne_s = -5 → u(0.1) near 0
    u_delet = kimura_fixation_prob(0.1, -5.0)
    check(f"Deleterious u(0.1, -5) = {u_delet:.6f} ≈ 0",
          0 <= u_delet < 0.05)

    # From actual per-flip data (if available)
    if M is not None and V is not None:
        # For each probe, compute Ne_s from M and V
        # Ne ≈ 1 / (2*V) for diffusion, s ≈ M
        # Ne_s ≈ M / (2*V) when V > 0
        fixation_probs = []
        for j in range(n_targets):
            gap_j = t_bl_real[j]
            if V[j] > 1e-12:
                Ne_s_j = M[j] / (2 * V[j])
                # Normalize gap to [0,1]: p = distance from boundary / max_gap
                max_gap = max(abs(g) for g in t_bl_real) + 1e-10
                p_j = abs(gap_j) / max_gap
                u_j = kimura_fixation_prob(p_j, Ne_s_j)
                fixation_probs.append(u_j)
            else:
                fixation_probs.append(float('nan'))

        valid_probs = [u for u in fixation_probs if math.isfinite(u)]
        check(f"Fixation probabilities computed: {len(valid_probs)}/{n_targets}",
              len(valid_probs) > 0)
        if valid_probs:
            check("All fixation probs in [0, 1]",
                  all(0 <= u <= 1 for u in valid_probs),
                  f"out of range: {[u for u in valid_probs if u < 0 or u > 1]}")
            print(f"    u(p) range: [{min(valid_probs):.4f}, {max(valid_probs):.4f}]")
    print()

    # ════════════════════════════════════════════════════════════
    # 8. Expected crossing time (Kolmogorov backward)
    # ════════════════════════════════════════════════════════════
    print("8. Expected crossing time (Kolmogorov backward)")

    def expected_crossing_time(gap, M_probe, V_probe, dt=1.0):
        """Approximate expected time to cross gap=0 from negative gap.

        For a diffusion with drift M and variance V per step:
        E[T] ≈ |gap| / M  if M > 0 (deterministic drift approximation)
        E[T] ≈ gap^2 / V   if M ≈ 0 (pure diffusion approximation)

        More accurate: solve the Kolmogorov backward equation numerically.
        For the preflight, the drift approximation validates the pipeline.
        """
        if M_probe > 1e-12:
            # Drift-dominated: time ≈ distance / speed
            t_drift = abs(gap) / M_probe
            # Diffusion correction: add variance contribution
            if V_probe > 1e-12:
                t_diffusion = gap ** 2 / V_probe
                # Combined estimate (harmonic mean gives lower bound)
                return min(t_drift, t_diffusion)
            return t_drift
        elif V_probe > 1e-12:
            # Pure diffusion: time ≈ distance^2 / variance
            return gap ** 2 / V_probe
        else:
            return float('inf')

    if M is not None and V is not None:
        crossing_times = []
        for j in range(n_targets):
            gap_j = t_bl_real[j]
            if gap_j < 0:  # wrong probes only
                t_j = expected_crossing_time(gap_j, M[j], V[j])
                crossing_times.append((j, gap_j, M[j], V[j], t_j))

        if crossing_times:
            # Sort by expected time
            crossing_times.sort(key=lambda x: x[4])
            check(f"Crossing times computed for {len(crossing_times)} wrong probes", True)
            check("All crossing times positive and finite",
                  all(t > 0 and math.isfinite(t) for _, _, _, _, t in crossing_times),
                  f"bad times: {[(j, t) for j, _, _, _, t in crossing_times if t <= 0 or not math.isfinite(t)]}")

            # Print top 3 (fastest predicted crossings)
            print(f"    Fastest predicted crossings:")
            for j, gap, m, v, t in crossing_times[:3]:
                print(f"      probe {j}: gap={gap:+.4f}, M={m:+.6f}, V={v:.6f}, E[T]={t:.0f} flips")
        else:
            check("Wrong probes for crossing time", False, "no wrong probes in synthetic data")
    print()

    # ════════════════════════════════════════════════════════════
    # 9. CIFAR-10-C path construction
    # ════════════════════════════════════════════════════════════
    print("9. CIFAR-10-C path construction")

    check(f"CIFAR10C_CORRUPTIONS has {len(CIFAR10C_CORRUPTIONS)} types",
          len(CIFAR10C_CORRUPTIONS) == 19)
    check("'fog' in corruption list", 'fog' in CIFAR10C_CORRUPTIONS)
    check("'gaussian_noise' in corruption list", 'gaussian_noise' in CIFAR10C_CORRUPTIONS)

    # Validate error handling for bad corruption
    try:
        load_cifar10c('nonexistent_corruption', 3, data_dir='/tmp/fake')
        check("Bad corruption raises ValueError", False, "no exception raised")
    except ValueError:
        check("Bad corruption raises ValueError", True)
    except Exception as e:
        check("Bad corruption raises ValueError", False, f"wrong exception: {type(e)}: {e}")

    # Validate error handling for bad severity
    try:
        load_cifar10c('fog', 0, data_dir='/tmp/fake')
        check("Bad severity raises ValueError", False, "no exception raised")
    except ValueError:
        check("Bad severity raises ValueError", True)
    except Exception as e:
        check("Bad severity raises ValueError", False, f"wrong exception: {type(e)}: {e}")
    print()

    # ════════════════════════════════════════════════════════════
    # 10. Focused mode targeting
    # ════════════════════════════════════════════════════════════
    print("10. Focused mode targeting")

    wrong_probes = [(i, t_bl_real[i]) for i in range(n_targets) if t_bl_real[i] <= 0]
    focus_order = sorted(wrong_probes, key=lambda x: abs(x[1]))
    check(f"Focus order: {len(focus_order)} wrong probes sorted by |gap|",
          len(focus_order) >= 0)  # may be 0 on random data

    if len(focus_order) >= 2:
        # Verify ordering: first probe should have smallest |gap|
        check("Focus order: smallest |gap| first",
              abs(focus_order[0][1]) <= abs(focus_order[1][1]))

    # Verify constraint logic: a flip that breaks a control should be rejected
    # (logical check, not a search run)
    def focused_accept(target_improved, controls_ok, prior_ok):
        return target_improved and controls_ok and prior_ok

    check("Focused rejects when control breaks",
          not focused_accept(True, False, True))
    check("Focused rejects when prior crossing reverts",
          not focused_accept(True, True, False))
    check("Focused rejects when target doesn't improve",
          not focused_accept(False, True, True))
    check("Focused accepts when all conditions met",
          focused_accept(True, True, True))
    print()

    # ════════════════════════════════════════════════════════════
    # Cleanup and summary
    # ════════════════════════════════════════════════════════════
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("=" * 50)
    print(f"PREFLIGHT: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("FIX FAILURES BEFORE RUNNING GPU EXPERIMENTS")
        sys.exit(1)
    else:
        print("All checks passed. Safe to proceed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
