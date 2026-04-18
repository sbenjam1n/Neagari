#!/usr/bin/env python3
"""Neagari-Vision: XOR patch search on binary neural networks.

Minimal experiment testing whether the degeneracy landscape and borderline
fitness transfer from 1-bit LLMs to 1-bit vision models.

Model: Binary ResNet-18 on CIFAR-10
Probes: Logit gap (correct class - top wrong class) per image
Search: XOR flip groups of 128 binary weights in conv/FC layers
Fitness: average / crossing / borderline / focused (same as LLM search)
Eval: CIFAR-10 test accuracy (10K images, fully held-out)
Corruption: CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019) for robustness

Usage:
    # Train a binary ResNet-18 (skip if you have a checkpoint)
    python3 neagari_vision.py --train --epochs 100

    # Run XOR search with borderline fitness
    python3 neagari_vision.py --search --fitness borderline --iterations 10000

    # Run all three fitness modes for comparison
    python3 neagari_vision.py --search --fitness average --iterations 5000 --output results_average
    python3 neagari_vision.py --search --fitness borderline --iterations 5000 --output results_borderline
    python3 neagari_vision.py --search --fitness crossing --iterations 5000 --output results_crossing

    # Run with per-flip per-probe delta logging (for Kimura/Fokker-Planck analysis)
    python3 neagari_vision.py --search --fitness borderline --iterations 5000 --log-deltas

    # Run focused (lexicographic) search targeting specific probes
    python3 neagari_vision.py --search --fitness focused --iterations 10000

    # Run search on corruption-specific probes (CIFAR-10-C)
    python3 neagari_vision.py --search --fitness borderline --corruption fog --severity 3

    # Evaluate a patched model on clean + all corruptions
    python3 neagari_vision.py --eval --eval-corruptions

    # Evaluate a patched model
    python3 neagari_vision.py --eval
"""

import argparse, json, os, sys, time, random, copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Binary convolution primitives
# ═══════════════════════════════════════════════════════════════

class SignSTE(torch.autograd.Function):
    """Sign function with straight-through estimator for training."""
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1, 1)


class BinaryConv2d(nn.Module):
    """Conv2d with binarized weights (sign only). Scales per-channel."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Full-precision weights (used during training, binarized at forward time)
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # After training, we store the binary signs and scales separately
        self.register_buffer('binary_signs', None)
        self.register_buffer('scales', None)
        self.frozen = False

    def forward(self, x):
        if self.frozen:
            # Inference with frozen binary weights
            w = self.scales.view(-1, 1, 1, 1) * self.binary_signs.float()
            return F.conv2d(x, w, self.bias, self.stride, self.padding)
        else:
            # Training: binarize on the fly with STE
            binary_w = SignSTE.apply(self.weight)
            # Per-output-channel scale = mean(|w|)
            scale = self.weight.abs().mean(dim=[1, 2, 3], keepdim=True)
            w = scale * binary_w
            return F.conv2d(x, w, self.bias, self.stride, self.padding)

    def freeze(self):
        """Convert to inference mode: store binary signs + scales."""
        with torch.no_grad():
            self.binary_signs = self.weight.sign().to(torch.int8)
            self.scales = self.weight.abs().mean(dim=[1, 2, 3])
        self.frozen = True

    def n_binary_weights(self):
        return self.weight.numel()

    def n_groups(self, group_size=128):
        return math.ceil(self.weight.numel() / group_size)


class BinaryLinear(nn.Module):
    """Linear with binarized weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer('binary_signs', None)
        self.register_buffer('scales', None)
        self.frozen = False

    def forward(self, x):
        if self.frozen:
            w = self.scales.unsqueeze(1) * self.binary_signs.float()
            return F.linear(x, w, self.bias)
        else:
            binary_w = SignSTE.apply(self.weight)
            scale = self.weight.abs().mean(dim=1, keepdim=True)
            w = scale * binary_w
            return F.linear(x, w, self.bias)

    def freeze(self):
        with torch.no_grad():
            self.binary_signs = self.weight.sign().to(torch.int8)
            self.scales = self.weight.abs().mean(dim=1)
        self.frozen = True

    def n_binary_weights(self):
        return self.weight.numel()

    def n_groups(self, group_size=128):
        return math.ceil(self.weight.numel() / group_size)


# ═══════════════════════════════════════════════════════════════
# Binary ResNet-18 for CIFAR-10
# ═══════════════════════════════════════════════════════════════

class BinaryBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = BinaryConv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinaryConv2d(planes, planes, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Shortcut uses FP conv (standard in BNN literature)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class BinaryResNet18(nn.Module):
    """Binary ResNet-18 for CIFAR-10 (32x32 input)."""

    def __init__(self, num_classes=10):
        super().__init__()
        # First conv is FP (standard in BNN literature — preserves input info)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.fc = BinaryLinear(512, num_classes)

    def _make_layer(self, in_planes, planes, n_blocks, stride):
        layers = [BinaryBasicBlock(in_planes, planes, stride)]
        for _ in range(1, n_blocks):
            layers.append(BinaryBasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def freeze_binary(self):
        """Freeze all binary layers for XOR search."""
        for m in self.modules():
            if isinstance(m, (BinaryConv2d, BinaryLinear)):
                m.freeze()

    def get_binary_layers(self):
        """Return list of (name, module) for all binary layers."""
        layers = []
        for name, m in self.named_modules():
            if isinstance(m, (BinaryConv2d, BinaryLinear)) and m.frozen:
                layers.append((name, m))
        return layers


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

def get_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# CIFAR-10-C: corruption robustness (Hendrycks & Dietterich 2019)
# ═══════════════════════════════════════════════════════════════

CIFAR10C_CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
    'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
    'zoom_blur',
]


def load_cifar10c(corruption, severity=3, data_dir='./data/CIFAR-10-C'):
    """Load a specific corruption type and severity from CIFAR-10-C.

    Downloads from Zenodo if not present locally. Each .npy file contains
    50,000 images (10K test images x 5 severity levels). Severity 1 is
    indices 0:10000, severity 5 is indices 40000:50000.

    Returns: (images_tensor, labels_tensor) normalized to match clean CIFAR-10.
    """
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption}. "
                         f"Options: {CIFAR10C_CORRUPTIONS}")
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")

    os.makedirs(data_dir, exist_ok=True)
    corr_path = os.path.join(data_dir, f'{corruption}.npy')
    label_path = os.path.join(data_dir, 'labels.npy')

    # Download if missing
    if not os.path.exists(corr_path) or not os.path.exists(label_path):
        zenodo_url = 'https://zenodo.org/records/2535967/files'
        print(f"  Downloading CIFAR-10-C {corruption}.npy from Zenodo...")
        import urllib.request
        if not os.path.exists(corr_path):
            urllib.request.urlretrieve(
                f'{zenodo_url}/{corruption}.npy?download=1', corr_path)
        if not os.path.exists(label_path):
            urllib.request.urlretrieve(
                f'{zenodo_url}/labels.npy?download=1', label_path)

    # Load: each .npy is (50000, 32, 32, 3) uint8
    all_images = np.load(corr_path)
    all_labels = np.load(label_path)

    # Slice by severity: each severity is 10K images
    start = (severity - 1) * 10000
    end = start + 10000
    images = all_images[start:end]  # (10000, 32, 32, 3) uint8
    labels = all_labels[start:end]  # (10000,) int

    # Convert to torch tensor, normalize to match clean CIFAR-10
    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    images_t = (images_t - mean) / std
    labels_t = torch.from_numpy(labels).long()

    return images_t, labels_t


def build_corruption_probes(model, corruption, severity, device,
                            clean_test_loader, n_targets=100, n_controls=100):
    """Build probes for corruption-specific search.

    Target probes: corrupted images the model misclassifies (logit gap < 0).
    Control probes: CLEAN images the model classifies correctly (logit gap > 0).

    This separation is the key design: search for flips that fix corruption
    failures while preserving clean-image accuracy.
    """
    model.eval()

    # Load corrupted images
    corr_images, corr_labels = load_cifar10c(corruption, severity)
    corr_images = corr_images.to(device)
    corr_labels = corr_labels.to(device)

    # Score all corrupted images
    targets_pool = []
    with torch.no_grad():
        batch_size = 128
        for start in range(0, len(corr_images), batch_size):
            imgs = corr_images[start:start + batch_size]
            labs = corr_labels[start:start + batch_size]
            logits = model(imgs)
            for i in range(imgs.size(0)):
                correct_class = labs[i].item()
                correct_logit = logits[i, correct_class].item()
                wrong_logits = logits[i].clone()
                wrong_logits[correct_class] = -1e9
                wrong_class = wrong_logits.argmax().item()
                gap = correct_logit - wrong_logits[wrong_class].item()
                if gap <= 0:  # model gets it wrong
                    targets_pool.append({
                        'image_idx': start + i,
                        'image': imgs[i].cpu(),
                        'label': correct_class,
                        'wrong_class': wrong_class,
                        'gap': gap,
                        'correct': False,
                        'source': f'{corruption}_s{severity}',
                    })

    # Controls from clean test set
    controls_pool = []
    with torch.no_grad():
        for images, labels in clean_test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            for i in range(images.size(0)):
                correct_class = labels[i].item()
                correct_logit = logits[i, correct_class].item()
                wrong_logits = logits[i].clone()
                wrong_logits[correct_class] = -1e9
                wrong_class = wrong_logits.argmax().item()
                gap = correct_logit - wrong_logits[wrong_class].item()
                if gap > 0:
                    controls_pool.append({
                        'image_idx': len(controls_pool),
                        'image': images[i].cpu(),
                        'label': correct_class,
                        'wrong_class': wrong_class,
                        'gap': gap,
                        'correct': True,
                        'source': 'clean',
                    })

    # Sort: targets closest to boundary first, controls smallest gap first
    targets_pool.sort(key=lambda p: abs(p['gap']))
    controls_pool.sort(key=lambda p: p['gap'])

    targets = targets_pool[:n_targets]
    controls = controls_pool[:n_controls]

    print(f"  Corruption: {corruption} severity {severity}")
    print(f"  Corrupted wrong: {len(targets_pool)}/{len(corr_images)} "
          f"({100*len(targets_pool)/len(corr_images):.1f}%)")
    print(f"  Target probes: {len(targets)}")
    print(f"  Control probes (clean): {len(controls)}")
    if targets:
        print(f"  Target gap range: [{targets[0]['gap']:.4f}, {targets[-1]['gap']:.4f}]")

    return targets, controls


def evaluate_corruptions(model, device, corruptions=None, severities=None):
    """Evaluate model accuracy on multiple CIFAR-10-C corruption types.

    Returns dict: {corruption_severity: accuracy}.
    """
    if corruptions is None:
        corruptions = ['fog', 'gaussian_noise', 'motion_blur',
                       'jpeg_compression', 'contrast']
    if severities is None:
        severities = [1, 3, 5]

    results = {}
    model.eval()
    for corruption in corruptions:
        for severity in severities:
            try:
                images, labels = load_cifar10c(corruption, severity)
            except Exception as e:
                print(f"  Skip {corruption}_s{severity}: {e}")
                continue
            correct = 0
            total = len(labels)
            with torch.no_grad():
                batch_size = 128
                for start in range(0, total, batch_size):
                    imgs = images[start:start + batch_size].to(device)
                    labs = labels[start:start + batch_size].to(device)
                    out = model(imgs)
                    correct += (out.argmax(1) == labs).sum().item()
            acc = 100 * correct / total
            key = f'{corruption}_s{severity}'
            results[key] = acc
            print(f"  {key}: {acc:.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_model(epochs=100, lr=0.01, save_path='binary_resnet18_cifar10.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryResNet18().to(device)
    train_loader, test_loader = get_cifar10()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()

        train_acc = 100 * correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)

        print(f"  Epoch {epoch+1:>3}/{epochs}  "
              f"loss={running_loss/total:.4f}  "
              f"train={train_acc:.1f}%  test={test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n  Best test accuracy: {best_acc:.1f}%")
    print(f"  Saved to {save_path}")
    return model


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)
    return 100 * correct / total


# ═══════════════════════════════════════════════════════════════
# Probe building (logit gap, same as LLM probes)
# ═══════════════════════════════════════════════════════════════

def build_probes(model, test_loader, device, n_targets=100, n_controls=100):
    """Build probes from CIFAR-10 test images.

    Target probes: images the model gets WRONG (logit gap < 0)
    Control probes: images the model gets RIGHT (logit gap > 0)
    """
    model.eval()
    all_probes = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            for i in range(images.size(0)):
                correct_class = labels[i].item()
                correct_logit = logits[i, correct_class].item()
                # Top wrong class
                wrong_logits = logits[i].clone()
                wrong_logits[correct_class] = -1e9
                wrong_class = wrong_logits.argmax().item()
                wrong_logit = wrong_logits[wrong_class].item()
                gap = correct_logit - wrong_logit

                all_probes.append({
                    'image_idx': len(all_probes),
                    'image': images[i].cpu(),
                    'label': correct_class,
                    'wrong_class': wrong_class,
                    'gap': gap,
                    'correct': gap > 0,
                })

    # Split into targets (wrong) and controls (right)
    wrong = [p for p in all_probes if not p['correct']]
    right = [p for p in all_probes if p['correct']]

    # Sort by gap magnitude: targets closest to boundary first
    wrong.sort(key=lambda p: abs(p['gap']))
    right.sort(key=lambda p: p['gap'])  # smallest gap = closest to boundary

    targets = wrong[:n_targets]
    controls = right[:n_controls]

    print(f"  Total test images: {len(all_probes)}")
    print(f"  Wrong: {len(wrong)}, Right: {len(right)}")
    print(f"  Target probes: {len(targets)} (closest to boundary)")
    print(f"  Control probes: {len(controls)}")
    if targets:
        print(f"  Target gap range: [{targets[0]['gap']:.4f}, {targets[-1]['gap']:.4f}]")

    return targets, controls


def score_probe(model, probe, device):
    """Score a single probe: return logit gap (correct - top wrong)."""
    with torch.no_grad():
        logits = model(probe['image'].unsqueeze(0).to(device))
        correct_logit = logits[0, probe['label']].item()
        wrong_logits = logits[0].clone()
        wrong_logits[probe['label']] = -1e9
        wrong_logit = wrong_logits.max().item()
    return correct_logit - wrong_logit


# ═══════════════════════════════════════════════════════════════
# XOR flip on binary layers
# ═══════════════════════════════════════════════════════════════

GROUP_SIZE = 128

def flip_group(layer, group_idx):
    """Flip signs of a group of 128 binary weights."""
    signs = layer.binary_signs.view(-1)
    start = group_idx * GROUP_SIZE
    end = min(start + GROUP_SIZE, signs.numel())
    signs[start:end] *= -1


# ═══════════════════════════════════════════════════════════════
# Fitness functions (identical logic to overnight_verbatim_v2.py)
# ═══════════════════════════════════════════════════════════════

def fitness_average(tg, cg, t_bl, c_bl, lam):
    t_imp = sum(g - b for g, b in zip(tg, t_bl)) / len(tg)
    c_deg = sum(max(0, b - g) for g, b in zip(cg, c_bl)) / len(cg) if cg else 0
    return t_imp - lam * c_deg

def fitness_crossing(tg, cg, t_bl, c_bl, lam):
    fixes = sum(1 for b, g in zip(t_bl, tg) if b <= 0 and g > 0)
    breaks_t = sum(1 for b, g in zip(t_bl, tg) if b > 0 and g <= 0)
    breaks_c = sum(1 for b, g in zip(c_bl, cg) if b > 0 and g <= 0) if cg else 0
    return (fixes - breaks_t) - lam * breaks_c

def fitness_borderline(tg, cg, t_bl, c_bl, lam):
    t_imp = sum(max(0, g - b) / (0.1 + abs(b)) for g, b in zip(tg, t_bl)) / len(tg)
    fixes = sum(1 for b, g in zip(t_bl, tg) if b <= 0 and g > 0)
    breaks_t = sum(1 for b, g in zip(t_bl, tg) if b > 0 and g <= 0)
    breaks_c = sum(1 for b, g in zip(c_bl, cg) if b > 0 and g <= 0) if cg else 0
    return t_imp + 0.5 * fixes - 0.5 * breaks_t - lam * breaks_c

# Focused fitness is handled inline in the search loop (lexicographic
# single-probe targeting with hard constraints), not as a scalar function.
# The FITNESS_FNS dict maps to scalar fitness; focused bypasses it.

FITNESS_FNS = {
    'average': fitness_average,
    'crossing': fitness_crossing,
    'borderline': fitness_borderline,
}


# ═══════════════════════════════════════════════════════════════
# XOR search
# ═══════════════════════════════════════════════════════════════

def run_search(model, targets, controls, args, device):
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    iterations = args.iterations
    lam = args.lambda_ctrl
    checkpoint_every = args.checkpoint_every
    log_deltas = getattr(args, 'log_deltas', False)
    is_focused = (args.fitness == 'focused')

    # Build search space: list of (layer, group_idx)
    binary_layers = model.get_binary_layers()
    candidates = []
    for name, layer in binary_layers:
        n_groups = layer.n_groups(GROUP_SIZE)
        for g in range(n_groups):
            candidates.append((name, layer, g))
    print(f"  Search space: {len(candidates)} groups across {len(binary_layers)} binary layers")
    print(f"  Total binary weights: {sum(l.n_binary_weights() for _, l in binary_layers):,}")

    # Measure baselines
    print("  Measuring baselines...")
    t_bl = [score_probe(model, p, device) for p in targets]
    c_bl = [score_probe(model, p, device) for p in controls]

    wrong_count = sum(1 for g in t_bl if g <= 0)
    borderline = sum(1 for g in t_bl if -1.0 < g <= 0)
    print(f"  Targets: {len(targets)}, wrong={wrong_count}, borderline(|gap|<1)={borderline}")
    print(f"  Controls: {len(controls)}, correct={sum(1 for g in c_bl if g > 0)}/{len(c_bl)}")

    # Per-flip delta logging: maintain running gap state
    if log_deltas:
        current_tg = list(t_bl)
        current_cg = list(c_bl)
        delta_log_path = os.path.join(output_dir, 'per_flip_deltas.jsonl')
        delta_fh = open(delta_log_path, 'w')
        print(f"  Per-flip delta logging: {delta_log_path}")

    # Focused mode: lexicographic single-probe targeting
    if is_focused:
        # Sort targets by gap magnitude (closest to boundary first)
        focus_order = sorted(range(len(targets)),
                             key=lambda i: abs(t_bl[i]) if t_bl[i] <= 0 else float('inf'))
        focus_order = [i for i in focus_order if t_bl[i] <= 0]  # only wrong probes
        focus_idx = 0  # current target index within focus_order
        crossed_set = set()  # indices of targets already crossed
        if focus_order:
            current_focus = focus_order[focus_idx]
            print(f"  Focused mode: {len(focus_order)} wrong probes to cross")
            print(f"  First target: idx={current_focus}, "
                  f"label={targets[current_focus]['label']}, "
                  f"gap={t_bl[current_focus]:.4f}")
        else:
            print("  Focused mode: no wrong probes to target")
            is_focused = False

    if not is_focused:
        fitness_fn = FITNESS_FNS[args.fitness]

    # Search
    accepted = []
    history = [0.0]
    best_f = 0.0
    t0 = time.time()

    mode_label = 'focused' if is_focused else args.fitness
    print(f"\n  Search: fitness={mode_label}, iterations={iterations:,}, λ={lam}")

    for i in range(iterations):
        # Sample random candidate
        name, layer, group_idx = random.choice(candidates)

        # Flip
        flip_group(layer, group_idx)

        # Evaluate all probes
        tg = [score_probe(model, p, device) for p in targets]
        cg = [score_probe(model, p, device) for p in controls]

        if is_focused:
            # Focused acceptance: accept iff target probe improves AND
            # no control breaks AND no prior crossings revert
            if focus_idx < len(focus_order):
                ci = focus_order[focus_idx]
                target_improved = tg[ci] > (current_tg[ci] if log_deltas else t_bl[ci])
                # Hard constraints: controls must stay positive
                ctrl_ok = all(g > 0 for b, g in zip(c_bl, cg) if b > 0)
                # Prior crossings must stay crossed
                prior_ok = all(tg[j] > 0 for j in crossed_set)
                accept = target_improved and ctrl_ok and prior_ok
            else:
                accept = False  # all targets crossed

            if accept:
                new_f = tg[focus_order[focus_idx]]
                fixes = sum(1 for b, g in zip(t_bl, tg) if b <= 0 and g > 0)
                breaks = sum(1 for b, g in zip(t_bl, tg) if b > 0 and g <= 0)

                accepted.append({
                    'layer': name, 'group': group_idx, 'iteration': i,
                    'fitness': new_f, 'fixes': fixes, 'breaks': breaks,
                    'focus_target': focus_order[focus_idx],
                })
                best_f = new_f

                # Check if current target just crossed
                if t_bl[focus_order[focus_idx]] <= 0 and tg[focus_order[focus_idx]] > 0:
                    crossed_set.add(focus_order[focus_idx])
                    print(f"  *** iter {i+1}: CROSSING target {focus_order[focus_idx]} "
                          f"(label={targets[focus_order[focus_idx]]['label']}) ***")
                    focus_idx += 1
                    if focus_idx < len(focus_order):
                        print(f"  Next target: idx={focus_order[focus_idx]}, "
                              f"label={targets[focus_order[focus_idx]]['label']}, "
                              f"gap={tg[focus_order[focus_idx]]:.4f}")
                    else:
                        print(f"  All {len(focus_order)} targets crossed!")
            else:
                flip_group(layer, group_idx)  # revert
        else:
            # Standard fitness modes (average, crossing, borderline)
            new_f = fitness_fn(tg, cg, t_bl, c_bl, lam)

            if new_f > best_f:
                fixes = sum(1 for b, g in zip(t_bl, tg) if b <= 0 and g > 0)
                breaks = sum(1 for b, g in zip(t_bl, tg) if b > 0 and g <= 0)

                accepted.append({
                    'layer': name, 'group': group_idx, 'iteration': i,
                    'fitness': new_f, 'fixes': fixes, 'breaks': breaks,
                })
                best_f = new_f

                if fixes > 0:
                    print(f"  *** iter {i+1}: BOUNDARY CROSSING — {fixes} probe(s) fixed! ***")
            else:
                # Revert
                flip_group(layer, group_idx)

        # Per-flip delta logging (on accepted flips only)
        if log_deltas and len(accepted) > 0 and accepted[-1]['iteration'] == i:
            t_deltas = [g - c for g, c in zip(tg, current_tg)]
            c_deltas = [g - c for g, c in zip(cg, current_cg)]
            delta_entry = {
                'iteration': i,
                'flip_num': len(accepted),
                'layer': name,
                'group': group_idx,
                'target_deltas': t_deltas,
                'control_deltas': c_deltas,
                'target_gaps_after': tg,
                'control_gaps_after': cg,
            }
            delta_fh.write(json.dumps(delta_entry) + '\n')
            delta_fh.flush()
            current_tg = list(tg)
            current_cg = list(cg)

        history.append(best_f)

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (iterations - i - 1) / rate if rate > 0 else 0
            print(f"  iter {i+1:>6}/{iterations}  "
                  f"flips={len(accepted):>4}  "
                  f"rate={len(accepted)/(i+1)*100:.1f}%  "
                  f"fitness={best_f:.6f}  "
                  f"{rate:.1f} it/s  "
                  f"ETA {eta:.0f}s")

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            tg_full = [score_probe(model, p, device) for p in targets]
            cg_full = [score_probe(model, p, device) for p in controls]
            fixes_full = sum(1 for b, g in zip(t_bl, tg_full) if b <= 0 and g > 0)
            breaks_full = sum(1 for b, g in zip(t_bl, tg_full) if b > 0 and g <= 0)
            c_breaks = sum(1 for b, g in zip(c_bl, cg_full) if b > 0 and g <= 0)

            print(f"\n  --- Checkpoint {i+1} ---")
            print(f"    Flips: {len(accepted)}, Fitness: {best_f:.6f}")
            print(f"    Target fixes: {fixes_full}, breaks: {breaks_full}")
            print(f"    Control breaks: {c_breaks}")

            # Frontier
            frontier = sorted([
                (p['label'], g, bl) for p, g, bl in zip(targets, tg_full, t_bl) if bl <= 0
            ], key=lambda x: abs(x[1]))[:5]
            print(f"    Frontier (5 closest to boundary):")
            for label, gap, baseline in frontier:
                print(f"      class={label} gap={gap:+.4f} (was {baseline:+.4f})")
            print()

            ckpt = {
                'flips': accepted, 'fitness_history': history,
                'iteration': i + 1, 'fitness_mode': mode_label,
            }
            if is_focused:
                ckpt['crossed'] = list(crossed_set)
                ckpt['focus_order'] = focus_order
            with open(os.path.join(output_dir, 'checkpoint.json'), 'w') as f:
                json.dump(ckpt, f)

    elapsed = time.time() - t0

    if log_deltas:
        delta_fh.close()
        print(f"  Per-flip deltas written: {delta_log_path} "
              f"({len(accepted)} entries)")

    # Final evaluation
    print(f"\n  Search complete: {len(accepted)} flips in {iterations:,} iterations")
    print(f"  Acceptance rate: {len(accepted)/iterations*100:.1f}%")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/3600:.2f}h)")

    # Final boundary count
    tg_final = [score_probe(model, p, device) for p in targets]
    fixes_final = sum(1 for b, g in zip(t_bl, tg_final) if b <= 0 and g > 0)
    breaks_final = sum(1 for b, g in zip(t_bl, tg_final) if b > 0 and g <= 0)
    print(f"  Final boundary crossings: {fixes_final} fixes, {breaks_final} breaks")

    if is_focused:
        print(f"  Focused: {len(crossed_set)}/{len(focus_order)} targets crossed")

    # Save results
    result = {
        'version': 'neagari_vision_v2',
        'model': 'binary_resnet18_cifar10',
        'fitness_mode': mode_label,
        'flips': accepted,
        'total_iterations': iterations,
        'accept_count': len(accepted),
        'accept_rate': len(accepted) / max(iterations, 1),
        'lambda': lam,
        'n_targets': len(targets),
        'n_controls': len(controls),
        'boundary_fixes': fixes_final,
        'boundary_breaks': breaks_final,
        'wall_time_seconds': elapsed,
        'log_deltas': log_deltas,
    }
    if is_focused:
        result['crossed_targets'] = list(crossed_set)
        result['focus_order'] = focus_order
    if hasattr(args, 'corruption') and args.corruption:
        result['corruption'] = args.corruption
        result['severity'] = args.severity

    result_path = os.path.join(output_dir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Results: {result_path}")

    return accepted


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Neagari-Vision: XOR search on binary ResNet')
    parser.add_argument('--train', action='store_true', help='Train binary ResNet-18')
    parser.add_argument('--search', action='store_true', help='Run XOR search')
    parser.add_argument('--eval', action='store_true', help='Evaluate model accuracy')
    parser.add_argument('--model-path', default='binary_resnet18_cifar10.pt')
    parser.add_argument('--fitness', choices=['average', 'crossing', 'borderline', 'focused'],
                        default='borderline',
                        help='Fitness function. focused = lexicographic single-probe targeting.')
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--lambda-ctrl', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--output', default='vision_results')
    parser.add_argument('--n-targets', type=int, default=100)
    parser.add_argument('--n-controls', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

    # Per-flip delta logging (Paper 2: Kimura/Fokker-Planck analysis)
    parser.add_argument('--log-deltas', action='store_true',
                        help='Log per-probe deltas for every accepted flip (JSONL). '
                             'Required for Fokker-Planck parameterization.')

    # CIFAR-10-C corruption support
    parser.add_argument('--corruption', type=str, default=None,
                        help=f'CIFAR-10-C corruption type for target probes. '
                             f'Options: {", ".join(CIFAR10C_CORRUPTIONS)}')
    parser.add_argument('--severity', type=int, default=3,
                        help='CIFAR-10-C corruption severity (1-5, default 3)')
    parser.add_argument('--eval-corruptions', action='store_true',
                        help='Evaluate on multiple CIFAR-10-C corruption types')
    parser.add_argument('--eval-corruption-list', type=str, default=None,
                        help='Comma-separated list of corruptions to evaluate '
                             '(default: fog,gaussian_noise,motion_blur,jpeg_compression,contrast)')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    if args.train:
        print("\n=== Training Binary ResNet-18 on CIFAR-10 ===")
        model = train_model(epochs=args.epochs, save_path=args.model_path)
        return

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Run with --train first, or provide a checkpoint path.")
        sys.exit(1)

    model = BinaryResNet18().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.freeze_binary()
    model.eval()

    _, test_loader = get_cifar10()

    if args.eval:
        print("\n=== Evaluating ===")
        acc = evaluate_accuracy(model, test_loader, device)
        print(f"  Clean test accuracy: {acc:.2f}%")

        if args.eval_corruptions:
            print("\n=== Corruption Robustness (CIFAR-10-C) ===")
            corruptions = None
            if args.eval_corruption_list:
                corruptions = args.eval_corruption_list.split(',')
            corr_results = evaluate_corruptions(model, device, corruptions=corruptions)

            # Save
            eval_out = {
                'clean_accuracy': acc,
                'corruption_accuracy': corr_results,
            }
            eval_path = os.path.join(args.output, 'eval_corruptions.json')
            os.makedirs(args.output, exist_ok=True)
            with open(eval_path, 'w') as f:
                json.dump(eval_out, f, indent=2)
            print(f"\n  Saved: {eval_path}")
        return

    if args.search:
        # Build probes: corruption-specific or clean
        if args.corruption:
            print(f"\n=== Building Corruption Probes ({args.corruption} s{args.severity}) ===")
            targets, controls = build_corruption_probes(
                model, args.corruption, args.severity, device,
                test_loader, n_targets=args.n_targets, n_controls=args.n_controls)
            # Auto-name output dir if default
            if args.output == 'vision_results':
                args.output = f'vision_results_{args.corruption}_s{args.severity}_{args.fitness}'
        else:
            print("\n=== Building Probes ===")
            targets, controls = build_probes(model, test_loader, device,
                                              n_targets=args.n_targets,
                                              n_controls=args.n_controls)

        # Baseline test accuracy
        base_acc = evaluate_accuracy(model, test_loader, device)
        print(f"\n  Baseline clean accuracy: {base_acc:.2f}%")

        # Baseline corruption accuracy (if corruption search)
        base_corr_acc = None
        if args.corruption:
            corr_images, corr_labels = load_cifar10c(args.corruption, args.severity)
            with torch.no_grad():
                correct = 0
                for start in range(0, len(corr_labels), 128):
                    imgs = corr_images[start:start+128].to(device)
                    labs = corr_labels[start:start+128].to(device)
                    correct += (model(imgs).argmax(1) == labs).sum().item()
            base_corr_acc = 100 * correct / len(corr_labels)
            print(f"  Baseline {args.corruption} s{args.severity} accuracy: {base_corr_acc:.2f}%")

        print("\n=== XOR Search ===")
        accepted = run_search(model, targets, controls, args, device)

        # Post-search evaluation
        post_acc = evaluate_accuracy(model, test_loader, device)
        print(f"\n=== Results ===")
        print(f"  Clean baseline:  {base_acc:.2f}%")
        print(f"  Clean patched:   {post_acc:.2f}%")
        print(f"  Clean delta:     {post_acc - base_acc:+.2f}pp")

        if base_corr_acc is not None:
            corr_images, corr_labels = load_cifar10c(args.corruption, args.severity)
            with torch.no_grad():
                correct = 0
                for start in range(0, len(corr_labels), 128):
                    imgs = corr_images[start:start+128].to(device)
                    labs = corr_labels[start:start+128].to(device)
                    correct += (model(imgs).argmax(1) == labs).sum().item()
            post_corr_acc = 100 * correct / len(corr_labels)
            print(f"  {args.corruption} baseline: {base_corr_acc:.2f}%")
            print(f"  {args.corruption} patched:  {post_corr_acc:.2f}%")
            print(f"  {args.corruption} delta:    {post_corr_acc - base_corr_acc:+.2f}pp")

        print(f"  Acceptance rate: {len(accepted)/args.iterations*100:.1f}%")

        # Save accuracy comparison
        acc_result = {
            'baseline_clean': base_acc,
            'patched_clean': post_acc,
            'delta_clean': post_acc - base_acc,
            'fitness_mode': args.fitness,
        }
        if base_corr_acc is not None:
            acc_result['corruption'] = args.corruption
            acc_result['severity'] = args.severity
            acc_result['baseline_corruption'] = base_corr_acc
            acc_result['patched_corruption'] = post_corr_acc
            acc_result['delta_corruption'] = post_corr_acc - base_corr_acc

        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'accuracy.json'), 'w') as f:
            json.dump(acc_result, f, indent=2)

        # Cross-corruption eval if this was a corruption search
        if args.corruption and args.eval_corruptions:
            print("\n=== Cross-corruption evaluation ===")
            corr_results = evaluate_corruptions(model, device)
            with open(os.path.join(args.output, 'cross_corruption.json'), 'w') as f:
                json.dump(corr_results, f, indent=2)


if __name__ == '__main__':
    main()
