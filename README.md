# SpecAudit-HSI

SpecAudit is a set of diagnostics for hyperspectral imagery processed by wavelength-conditioned foundation models, i.e. models whose patch-embedding entry generates one kernel per band from that band's wavelength (DOFA-style entries). It asks where the spectral evidence in a hyperspectral cube goes on its way through such a model, and measures this at three levels:

| Instrument          | Level          | Question                                                     | Needs                                                        |
| ------------------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Spectral Ledger** | Input          | Do the additional bands carry foreground/background evidence that a linear readout can extract, and is that evidence shared across images or specific to each image? | Pixel spectra and labels                                     |
| **KernelScope**     | Entry          | What does the pretrained entry do with the C bands it receives? | Frozen entry weights and a wavelength table (no images, no labels) |
| **DepthProbe**      | Representation | Is the evidence still linearly readable at each depth of the network, and what does task adaptation change? | Intermediate tokens and patch labels                         |

None of the readings update any parameter of the analysed network. The Fisher directions and PCA bases they use are estimated from data.

## The three instruments

### Spectral Ledger

All readings use one regularised Fisher discriminant, `w = (S_w + εI)^{-1} (μ_fg − μ_bg)` with ε = 10⁻³, whose projection is passed through a logistic function to give a saliency map. The Ledger compares a C-band configuration with a 3-band reference arm on the same pixels (true colour by default; any three bands can be specified), and varies only the scope over which the discriminant is fitted:

- **per-image, in-sample**: fitted and evaluated on the pixels of each test image;
- **per-image, split-half**: fitted on one half of an image's pixels and evaluated on the other half, and vice versa;
- **global**: fitted once on training-split pixels and applied to every test image without using its labels.

The difference between the per-image and global readings indicates how much of the additional evidence varies from image to image. Maps are scored with the S-measure of the official HRSSD evaluator and with rank-based AUC; differences are paired per image, with bootstrap confidence intervals.

Reliability controls: spatial split-half (left/right and top/bottom), a fixed global direction with per-image calibration (to separate direction from threshold), calibration-free AUC, and sensitivity to ε. A multi-class version (shared within-class covariance, mIoU) applies the same three readings to semantic segmentation.

### KernelScope

For an entry that computes `tokens = Σ_b K_b ∗ x_b`, the C generated kernels are stacked row-wise into a matrix K (C × 16·16·768 for DOFA ViT-B). KernelScope reports three quantities:

| Quantity             | Definition                                                   | Identical kernels | Mutually orthogonal kernels |
| -------------------- | ------------------------------------------------------------ | :---------------: | :-------------------------: |
| Mean pairwise cosine | Mean off-diagonal entry of the row-normalised Gram matrix    |         1         |              0              |
| Effective rank r_eff | `exp(−Σ p_i log p_i)`, with `p_i = σ_i / Σ σ` over the singular values of K |         1         |              C              |
| Collapse ratio χ     | `C ‖K̄‖² / ‖K‖_F²`, with K̄ the mean kernel                    |         1         |             1/C             |

The readings depend only on the frozen weights and the wavelengths, so any sensor can be analysed from its band table. Included: a table of multispectral and hyperspectral sensors, a band-density sweep over a fixed spectral range, arbitrary band subsets of one sensor, and an i.i.d. Gaussian reference of the same shape.

### DepthProbe

DepthProbe applies the Ledger's readouts to the tokens at each depth of the backbone instead of to pixel spectra:

- patch labels: a patch is foreground if more than half of its 16 × 16 label window is foreground;
- features: tokens reduced by PCA (k = 32 by default, k = 8 as a sensitivity check), fitted on training-split tokens separately for each arm and depth;
- readouts: per-image in-sample, per-image split-half and global Fisher readouts, with a ridge term relative to the trace of `S_w`;
- metric: rank-based AUC, so that no calibration is involved; per-image readings use images with at least 8 foreground and 8 background patches.

Two modes are provided: the **frozen** pretrained backbone, and a **task-adapted** backbone loaded from a checkpoint.

## What the readings do and do not show

- Per-image Ledger readings use the labels of the image being scored. They show whether linearly separable evidence exists; they do not estimate achievable accuracy and cannot be deployed as a predictor. Report the split-half reading, and preferably the spatial split-half, alongside the in-sample one.
- All readings measure separability under one specified linear readout, not information content.
- KernelScope describes the entry only and does not know which bands matter for a task. Kernel geometry is not a monotonic predictor of accuracy, and r_eff is an entropy index of the singular-value spectrum, not a count of preserved physical dimensions.
- DepthProbe measures the linear readability of PCA-reduced tokens and has no independent ground truth. Readings directly after the entry depend on the PCA truncation.

## Planned repository layout

```
SpecAudit-HSI/
├── specaudit/
│   ├── ledger/
│   │   ├── readings.py        # in-sample / split-half / global readings, C bands vs a 3-band reference arm
│   │   ├── checks.py          # reliability controls
│   │   └── segmentation.py    # multi-class readings for semantic segmentation
│   ├── kernelscope/
│   │   ├── sensors.py         # sensor table, band-density sweep, Gaussian reference
│   │   └── subsets.py         # band subsets of one sensor
│   └── depthprobe/
│       ├── frozen.py          # frozen pretrained backbone
│       └── adapted.py         # task-adapted backbone (DOFA ViT-B + LoRA)
├── third_party/               # DOFA entry and HRSSD loader/evaluator, with their licenses
├── requirements.txt
├── LICENSE
└── README.md
```

Each script writes its readings as a Markdown table and a CSV file.

## Requirements

Tested with Python 3.9, PyTorch 2.8.0 + CUDA 12.8, NumPy 2.0.2, SciPy 1.13.1, pandas, tifffile and GDAL (`conda install -c conda-forge gdal`). KernelScope runs on a CPU. The Spectral Ledger (through the S-measure evaluator) and DepthProbe expect a CUDA GPU.

## Data and pretrained weights

Datasets and weights are not redistributed. Obtain them from their original sources:

| Resource           | Source                                 | Notes                                                        |
| ------------------ | -------------------------------------- | ------------------------------------------------------------ |
| HRSSD              | https://github.com/laprf/HRSSD         | Official split `tr` / `ts`, 512 × 512 × 32 GeoTIFF; label value −1 marks pixels excluded from evaluation |
| WHU-OHS            | https://doi.org/10.5281/zenodo.7258035 | The segmentation readings use the official validation split (513 patches), re-split with a fixed seed |
| DOFA ViT-B weights | https://huggingface.co/earthflow/DOFA  | File `DOFA_ViT_base_e100.pth`, sha256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`. Later DOFA weights use a different entry implementation |

## Not included

Datasets, pretrained weights, trained checkpoints, experiment result files, and code for other published methods.

## License

The code is released under the MIT License (see `LICENSE`). Files in `third_party/` keep their original licenses:

| Component                                   | Source                           | License |
| ------------------------------------------- | -------------------------------- | ------- |
| DOFA wavelength-conditioned patch embedding | https://github.com/zhu-xlab/DOFA | MIT     |
| HRSSD data loader and S-measure evaluator   | https://github.com/laprf/HRSSD   | MIT     |

The DOFA pretrained weights are distributed by their authors under CC-BY-4.0.

## Citation

Citation information will be added after the paper is published.
