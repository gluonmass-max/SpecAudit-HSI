# SpecAudit-HSI

This repository provides the **Two-Level Spectral Adaptation (TLSA)** salient-object detector and the **SpecAudit** spectral diagnostics accompanying the [Remote Sensing paper](https://doi.org/10.3390/rs18213694). TLSA uses Band Amplitude Calibration (BAC) with a frozen DOFA ViT-B backbone, LoRA and a decoder, followed by training-free Self-Guided Spectral Refinement (SGSR) at inference.

SpecAudit is a set of diagnostics for hyperspectral imagery processed by wavelength-conditioned foundation models, i.e. models whose patch-embedding entry generates one kernel per band from that band's wavelength (DOFA-style entries). It asks where the spectral evidence in a hyperspectral cube goes on its way through such a model, and measures this at three levels:

| Instrument          | Level          | Question                                                     | Needs                                                        |
| ------------------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Spectral Ledger** | Input          | Do the additional bands carry foreground/background evidence that a linear readout can extract, and is that evidence shared across images or specific to each image? | Pixel spectra and labels                                     |
| **KernelScope**     | Entry          | What does the pretrained entry do with the C bands it receives? | Frozen entry weights and a wavelength table (no images, no labels) |
| **DepthProbe**      | Representation | Is the evidence still linearly readable at each depth of the network, and what does task adaptation change? | Intermediate tokens and patch labels                         |

None of the readings update any parameter of the analysed network. The Fisher directions and PCA bases they use are estimated from data.

## The three instruments

### Spectral Ledger

All readings use one regularised Fisher discriminant, `w = (S_w + εI)^{-1} (μ_fg − μ_bg)` with ε = 10⁻³, whose projection is passed through a logistic function to give a saliency map. The Ledger compares the full 32-band configuration with one of three predefined 3-band reference arms (`visible`, `random`, or `spread`; `visible` by default), and varies only the scope over which the discriminant is fitted:

- **per-image, in-sample**: fitted and evaluated on the pixels of each test image;
- **per-image, split-half**: fitted on one half of an image's pixels and evaluated on the other half, and vice versa;
- **global**: fitted once on training-split pixels and applied to every test image without using its labels.

The difference between the per-image and global readings indicates how much of the additional evidence varies from image to image. Maps are scored with the S-measure of the official HRSSD evaluator and with rank-based AUC; differences are paired per image, with bootstrap confidence intervals.

Implemented reliability controls include spatial split-half (left/right and top/bottom), fixed global directions with either label-free or label-calibrated per-image calibration, rank-based AUC, and a configurable regularisation strength (`--eps`). The released script covers binary foreground/background saliency on HRSSD; multi-class semantic-segmentation diagnostics are not included.

### KernelScope

For an entry that computes `tokens = Σ_b K_b ∗ x_b`, the C generated kernels are stacked row-wise into a matrix K (C × 16·16·768 for DOFA ViT-B). KernelScope reports three quantities:

| Quantity             | Definition                                                   | Identical kernels | Mutually orthogonal kernels |
| -------------------- | ------------------------------------------------------------ | :---------------: | :-------------------------: |
| Mean pairwise cosine | Mean off-diagonal entry of the row-normalised Gram matrix    |         1         |              0              |
| Effective rank r_eff | `exp(−Σ p_i log p_i)`, with `p_i = σ_i / Σ σ` over the singular values of K |         1         |              C              |
| Collapse ratio χ     | `C ‖K̄‖² / ‖K‖_F²`, with K̄ the mean kernel                    |         1         |             1/C             |

The current script evaluates the predefined band configurations in `tlsa/bands.py` using either the entry wavelengths (`--table entry`, default) or nominal image wavelengths (`--table image`), and prints an i.i.d. Gaussian reference of matching shape. A general sensor catalogue and automated band-density sweep are not part of this release.

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

## Repository layout

```text
SpecAudit-HSI/
├── train.py                     # train TLSA (BAC + LoRA + decoder)
├── infer.py                     # coarse maps and SGSR-refined maps
├── tlsa/
│   ├── model.py                 # TLSANet, spectral gating and decoder
│   ├── dofa_backbone.py         # DOFA ViT-B and LoRA integration
│   ├── wave_dynamic_layer.py    # DOFA wavelength-conditioned entry
│   ├── sgsr.py                  # training-free spectral refinement
│   ├── bands.py                 # wavelength tables and band subsets
│   └── data.py, losses.py, metrics.py
├── specaudit/
│   ├── ledger.py                # input-level spectral evidence
│   ├── kernelscope.py           # entry-kernel geometry
│   └── depthprobe.py            # depth-wise linear readability
├── eval/
│   ├── main.py                  # HRSSD evaluator entry point
│   └── evaluator.py, dataloader.py
├── third_party_licenses/
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Python 3.9 and the dependencies in `requirements.txt` were used for this release (including PyTorch 2.8.0 and NumPy 2.0.2). Install GDAL separately through conda. Training and inference call CUDA directly; KernelScope is CPU-only, while Ledger and DepthProbe can run without a GPU subject to available resources.

```bash
conda create -n tlsa python=3.9 -y
conda activate tlsa
conda install -c conda-forge gdal
pip install -r requirements.txt
```

## TLSA training, inference and evaluation

Train one seed (the paper uses seeds 7, 1, 2, 3 and 4):

```bash
python train.py --data_root HRSSD --dofa_ckpt DOFA_ViT_base_e100.pth --save_dir runs/tlsa_s7 --seed 7
```

The training script uses 100 epochs, batch size 16, NAdam with initial learning rate `3e-3`, cosine annealing, and BCE + IoU loss. A fixed 49-image validation subset is drawn from the official training split; the best checkpoint is written as `runs/tlsa_s7/model-best-val`.

Generate coarse and SGSR-refined saliency maps on the HRSSD test split:

```bash
python infer.py --data_root HRSSD --dofa_ckpt DOFA_ViT_base_e100.pth --ckpt runs/tlsa_s7/model-best-val --out_dir preds/tlsa_s7
```

The outputs are `preds/tlsa_s7/coarse/*.jpg` and `preds/tlsa_s7/tlsa/*.png`. Evaluate the refined maps using the included HRSSD evaluator:

```bash
python eval/main.py --method TLSA --root_dir preds/tlsa_s7/tlsa --gt_dir HRSSD/ts/label
```

## Run SpecAudit

```bash
python specaudit/ledger.py --data_root HRSSD --out results/ledger
python specaudit/kernelscope.py --dofa_ckpt DOFA_ViT_base_e100.pth
python specaudit/depthprobe.py --data_root HRSSD --dofa_ckpt DOFA_ViT_base_e100.pth --out results/depthprobe
```

- `ledger.py` saves per-image readings to CSV and a paired-difference summary to Markdown (`--ref visible|random|spread`, default `visible`).
- `kernelscope.py` prints its configuration table to standard output (`--table image` selects nominal HRSSD band centres).
- `depthprobe.py` saves depth-wise AUC differences and bootstrap intervals to CSV; its default mode is frozen DOFA. The optional adapted mode requires **both** compatible 32-band and 3-band TLSA checkpoints (`--ckpt_32b` and `--ckpt_3b`).

## Wavelength tables

`tlsa/bands.py` defines `ENTRY_WAVELENGTHS_NM` (the default supplied to the DOFA entry in the reported experiments) and `IMAGE_BAND_CENTRES_NM` (nominal HRSSD image-band centres). Keep the default entry table to reproduce the paper; KernelScope can compare the alternate table with `--table image`.

## Data and pretrained weights

Datasets and pretrained weights are not redistributed. Obtain them from their original sources:

| Resource | Source | Notes |
|---|---|---|
| HRSSD | https://github.com/laprf/HRSSD | Expected `HRSSD/{tr,ts}/{image,label,mask}/`: 32-band `.tif` images, `.tif` labels and `.png` masks. Label value `-1` denotes ignored pixels. |
| DOFA ViT-B weights | https://huggingface.co/earthflow/DOFA | Use `DOFA_ViT_base_e100.pth` (SHA-256: `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`). |

## Not included

Datasets, pretrained weights, trained checkpoints, experiment result files, multi-class WHU-OHS diagnostic scripts, and code for other published methods.

## License

The code is released under the MIT License (see `LICENSE`). Third-party components are acknowledged in `third_party_licenses/` (including `tlsa/wave_dynamic_layer.py` and HRSSD-derived loading/evaluation code):

| Component                                   | Source                           | License |
| ------------------------------------------- | -------------------------------- | ------- |
| DOFA wavelength-conditioned patch embedding | https://github.com/zhu-xlab/DOFA | MIT     |
| HRSSD data loader and S-measure evaluator   | https://github.com/laprf/HRSSD   | MIT     |

The DOFA pretrained weights are distributed by their authors under CC-BY-4.0.

## Citation

If you use TLSA or SpecAudit-HSI in your research, please cite the following paper:

Wang, S.; Tao, Y.; Cai, C.; Xu, H.; Zhang, Y.; Hong, Y. Two-Level Spectral Adaptation and Multilevel Spectral Evidence Diagnostics for Hyperspectral Remote Sensing Salient Object Detection with a Wavelength-Conditioned Foundation Model. *Remote Sensing* **2026**, *18*(21), 3694. https://doi.org/10.3390/rs18213694

```bibtex
@article{wang2026tlsa,
  author  = {Wang, Shentao and Tao, Ye and Cai, Chengkai and Xu, Haotian and Zhang, Yuanzhi and Hong, Yue},
  title   = {{Two-Level Spectral Adaptation and Multilevel Spectral Evidence Diagnostics for Hyperspectral Remote Sensing Salient Object Detection with a Wavelength-Conditioned Foundation Model}},
  journal = {Remote Sensing},
  year    = {2026},
  volume  = {18},
  number  = {21},
  pages   = {3694},
  doi     = {10.3390/rs18213694},
  url     = {https://doi.org/10.3390/rs18213694}
}
```
