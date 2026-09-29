# A Category-Aware Dual-Track Framework for Multi-View Industrial Anomaly Detection

**Alice7371**
*September 2026*

---

## Abstract

We present the system design behind our entry to the *Complex Industrial Scene
Anomaly Detection* track (CSIG 7th Image & Graphics Technology Challenge, Tianchi
raceId 231601). The task requires, for every industrial part captured by a
5-camera array at ~3589×3589 px, (i) image-level defect classification, (ii)
pixel-level segmentation into 448×448 masks, and (iii) zero-shot generalization
to 50 product categories never seen during training. Our method is
**category-aware and dual-track**: at inference the category is known, so seen
categories are routed to **PatchCore-style per-category memory banks** built on
frozen DINOv3-L/16 features with multi-scale zoom tiling, while unseen
categories are routed to **MuSc-style leave-one-out mutual scoring**, which
requires no normal references at all. Image scores are combined by rank fusion
across backbones, and anomaly masks receive **source-conditional Gaussian
smoothing** (σ chosen separately for bank-derived and mutual-scoring masks).
A Dinomaly model trained from the official data serves as a third fusion member.
On the official leaderboards the system reached **77.477 on Test A** and
**74.778 on Test B**, where the two largest single gains were MuSc for the
unseen track (+8.3 points) and source-conditional mask smoothing (+1.7 points).

## 1. Introduction

The challenge evaluates three objectives over the Real-IAD Variety corpus
(deep-cleaned subset):

1. **Image-level classification** — is the part defective? (anomaly probability)
2. **Pixel-level segmentation** — one 448×448 mask per view.
3. **Zero-shot generalization** — Test B adds 50 unseen categories.

Each sample is one part shot by a 5-camera array (`Sxxxx/0.png…4.png`,
~3589×3589 px on a white studio background). Training provides **normal samples
only**: 50 categories × 20 samples × 5 views. Defects (scratches, dents,
breakage, stains) are tiny relative to the frame — a 10 px scratch at 3589² is
≈1 px after naive downsampling — which makes segmentation the hardest and
highest-weighted objective.

The B-board score is

```
S = 100 × (0.3 · S_cls + 0.5 · S_seg + 0.2 · S_zs)
```

with macro-averaged AUROC/AP (image) and AUROC/AUPR/F1max (pixel); `S_zs`
averages image- and pixel-level metrics over the unseen categories. Inference
must stay under 1 s per sample. Notably, we recovered the A-board weighting
(≈ 0.3 image + 0.7 pixel) from submission feedback alone, before it mattered
for Test B.

## 2. Method

```
                       ┌──────────────────────────────────────────────┐
 test sample ──► category known?
                       ├─ seen ────► Track A: per-category memory bank
                       │             DINOv3-L/16 frozen backbone
                       │             448² global + 2×2 zoom tiles (5 passes/view)
                       │             score = max kNN-dist ratio over 5 views
                       │
                       └─ unseen ──► Track B: MuSc leave-one-out scoring
                                     (test set itself is the reference)
```

### 2.1 Track A — multi-scale per-category memory banks (seen)

- **Features.** DINOv3-L/16, frozen; patches from the last 3 blocks averaged,
  giving 784×1024-dim patch embeddings per pass.
- **Multi-scale passes.** One global 448² pass plus a 2×2 overlapping tile grid
  over the foreground bounding box, restoring ≈1800 px effective resolution
  where defects live. A 3×3 grid was rejected: slicing elongated parts destroys
  context and hard `max` fusion of tiles produces seam artifacts.
- **Bank & scoring.** Per-category bank of 60k patches (fp16, reservoir-sampled
  from the 20×5 normal views). Image score = max over patches of
  `kNN distance / bank median`, then max over the 5 views.

### 2.2 Track B — leave-one-out mutual scoring (unseen)

For unseen categories, banks are impossible by construction. We use MuSc-style
mutual scoring: all test samples of the category form the reference set, and
each patch is scored by its kNN distance to the *other* samples' patches, with
per-pass foreground-median normalization. This single change lifted the B board
from 64.8 to 73.1 — the zero-shot term had effectively collapsed to zero
without it.

### 2.3 Score fusion

Seen-category image scores are the **rank average** of a ViT-B/14-bank score
and the DINOv3-L/16 score; a 50/50 rank mix measured optimal, and a dense
fusion-weight sweep confirmed no further gain. Masks always come from the
DINOv3 track.

### 2.4 Source-conditional mask smoothing

Predicted masks are upsampled 32×32 grids and blocky, while ground-truth masks
are smooth blobs. Smoothing *strength must depend on the mask's source*:
bank-derived masks are tight (σ=4) while mutual-scoring masks are diffuse
(σ=11). Applying one global σ loses most of the benefit; per-source smoothing
added **+1.7 B-board points**.

### 2.5 Dinomaly as a third member

A Dinomaly model (reconstruction-based, trained on the official data only)
provides an independent image score and mask. Under a corrected fusion gate —
baseline = the *live 2-way incumbent*, PASS requires ≥ +0.6 pt macro I-AUROC
*and* paired bootstrap P(Δ>0) ≥ 0.95 — the 3-way fusion passed with **+1.68 pt**
on internal validation, using macro (not pooled) metrics throughout. This was
prepared as the next package but not submitted.

## 3. Results

Official leaderboard progression (best packages; A allows 3 submissions/day,
B allows 1):

| Board | Configuration | Score |
|---|---|---|
| A | ViT-B/14 banks (v1) | 76.679 |
| A | DINOv3-L banks | 77.106 |
| **A** | **+ rank fusion (ViT-B + DINOv3)** | **77.477** |
| B | ViT-B/14 banks (v1) | 64.789 |
| B | + MuSc for unseen (v2) | 73.113 |
| **B** | **+ per-source mask smoothing (v3)** | **74.778** |

### Negative results

All falsified internally at zero submission cost:

- **Bank augmentation** (hflip/rot90 into the bank): pollutes the bank and
  hurts image-level detection.
- **Larger global bank** (100k patches): bank size was not the unseen
  bottleneck — the routing was.
- **Synthetic-defect validation**: correlates poorly with real defects for
  zero-shot evaluation; only useful for pipeline correctness checks.
- **Mean-feature shift** for background suppression: redundant with the
  foreground mask (Δ ≤ 0.03 pt).
- **Transductive score calibration**: mathematically a no-op for macro rank
  metrics.
- **bf16 vs fp16 vs fp32** features: identical scores (Turing has no bf16;
  Ampere bf16 = fp16 throughput).

## 4. Repository Layout

```
ad/
  common.py            model load, DINOv3/v2 features, fg-mask, tiling, kNN, concurrent decode
  build_banks.py       per-category + global patch memory banks (reservoir sampling)
  predict.py           dual-track inference (seen banks / MuSc unseen), submission writer
  eval_valset.py       macro metrics evaluator (I-AUROC, P-AUROC, P-AUPR, P-F1max)
  validate_sub.py      submission format checker
  merge_subs.py        merge multi-part prediction folders
  fuse_rank.py         image-score rank fusion
  sigma_sweep.py       mask smoothing sweep + re-evaluation
  build_v3.py          final assembly: per-source mask smoothing on Test_B
  exp_fusion3.py       3-way fusion gate (ViT-B + DINOv3 + Dinomaly) + pixel-map fusion
  build_dinomaly_layout.py  official Train → Dinomaly data layout
  dinomaly_infer.py    Dinomaly inference → submission folder
dinomaly/              vendored upstream Dinomaly (Apache-2.0, a61959d)
  dinomaly_ours.py     training (resume-capable via --start_iter)
```

## 5. Reproduction

```bash
# python 3.12, torch (cu124), timm, opencv, scikit-learn; weights auto-download from HF
# 1) memory banks from the official training set
python ad/build_banks.py --train data/Train --out work/banks_D3 \
    --model vit_large_patch16_dinov3.lvd1689m

# 2) dual-track inference + submission (banks for seen, MuSc for unseen)
python ad/predict.py --test data/Test_B --out work/sub_B \
    --banks work/banks_D3 --model vit_large_patch16_dinov3.lvd1689m --musc

# 3) format check
python ad/validate_sub.py --test data/Test_B --sub work/sub_B

# 4) optional: per-source mask smoothing (σ=4 banks / σ=11 MuSc), see build_v3.py

# 5) Dinomaly third member
python ad/build_dinomaly_layout.py --train data/Train --root dinomaly_data
python dinomaly/dinomaly_ours.py --iters 50000 --save_every 10000 --save_name dinomaly_uni_ours
python dinomaly/dinomaly_infer.py --ckpt saved_results/dinomaly_uni_ours/model_50000.pth \
    --test data/val --out work/dino_infer_val   # then ad/exp_fusion3.py
```

## References

1. Roth et al. *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
2. Li et al. *MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring*, ICLR 2024.
3. Li et al. *Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection*, CVPR 2025. (vendored under `dinomaly/`, Apache-2.0)
4. Siméoni et al. *DINOv3*, 2025.

---

*Code shared for reference after the competition. The `dinomaly/` directory
vendors upstream [Dinomaly](https://github.com/guojiajeremy/Dinomaly) (Apache-2.0);
all other code written for this project.*
