# GAN-ATCN for Remaining Useful Life Prediction under Data Scarcity

A research implementation of a **conditional Generative Adversarial Network (cGAN)** combined with an **Attention-based Temporal Convolutional Network (ATCN)** for Remaining Useful Life (RUL) prediction under limited-data conditions.

The public pipeline currently focuses on the **NASA C-MAPSS turbofan benchmark**. It simulates data scarcity by retaining a fraction of the available training units, generates additional condition-aware sequences with a lightweight cGAN, and compares an ATCN trained on limited real data against an ATCN trained on real plus synthetic data.

## Research Motivation

High-reliability industrial systems often operate under preventive-maintenance policies and bounded data-retention infrastructures. As a result, complete run-to-failure histories and rare degradation states may be missing from the training distribution.

This project studies whether generative augmentation can reconstruct underrepresented health-condition coverage and improve prognostic performance without evaluating on synthetic test samples.

## Current Repository Scope

The current public release contains the C-MAPSS benchmark pipeline implemented in:

```text
GANs-ATCN.py
```

It supports the four standard C-MAPSS subsets:

- `FD001`
- `FD002`
- `FD003`
- `FD004`

The industrial ST07 SCARA case study described in the associated research work is not included in this repository because the raw production data are confidential.

## Pipeline

The script performs the following steps:

1. Load a selected C-MAPSS subset.
2. Construct piecewise-capped RUL labels.
3. Select degradation-relevant sensors and remove near-constant channels.
4. Build 50-cycle sliding windows.
5. Simulate a low-data regime by retaining 30% of the training units.
6. Fit a Min-Max scaler using only the reduced training split.
7. Train a conditional GAN on limited real sequences.
8. Generate synthetic sequences conditioned on RUL.
9. Train an ATCN on limited real data only.
10. Train a second ATCN on real plus synthetic data.
11. Evaluate both models exclusively on the real C-MAPSS test set.
12. Save trained models, metrics, CSV summaries, and diagnostic plots.

## Model Overview

### Conditional GAN

The generator maps latent noise and a conditioning RUL value to a multivariate sequence. The discriminator receives a sequence-RUL pair and predicts whether the sequence is real or generated.

Main settings in the current script:

- Input window: 50 cycles
- Latent dimension: 32
- GAN epochs: 40
- GAN batch size: 64
- Optimizer: Adam
- Generator output activation: `tanh`

### Attention-TCN

The forecasting network combines:

- Dense input projection
- Multi-head self-attention
- Four causal TCN blocks with dilation rates `1, 2, 4, 8`
- Residual connections
- Squeeze-and-Excitation channel recalibration
- Global average pooling
- Linear RUL regression output

Main settings:

- Model dimension: 64
- TCN filters: 64
- ATCN epochs: 30
- Batch size: 256
- Huber loss
- Early stopping
- Reduce-on-plateau learning-rate scheduling

## Repository Structure

```text
GAN-ATCN/
├── GANs-ATCN.py
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── ...
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
└── results_gan_atcn/
    └── FD00X/
        ├── generator_FD00X.h5
        ├── discriminator_FD00X.h5
        ├── atcn_real_small_FD00X.h5
        ├── atcn_real_plus_syn_FD00X.h5
        ├── real_vs_synthetic_sequences.png
        ├── pred_vs_true_atcn_real_vs_syn.png
        ├── scatter_atcn_real_vs_syn.png
        ├── residuals_atcn_real_vs_syn.png
        └── fd00x_gan_atcn_summary.csv
```

The `data/` and `results_gan_atcn/` directories are not currently included automatically. Create them locally as described below.

## Dataset

Download the NASA C-MAPSS jet-engine degradation dataset from the NASA Open Data Portal:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Place the extracted files in a local `data/` directory using the standard names expected by the script:
*

```text
data/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
├── RUL_FD003.txt
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/zerotwoo2-k/GAN-ATCN.git
cd GAN-ATCN
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required Python packages:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

## Usage

Open `GANs-ATCN.py` and choose the benchmark subset:

```python
FDX = "FD004"
```

Available values are:

```python
"FD001"
"FD002"
"FD003"
"FD004"
```

Run the complete experiment:

```bash
python GANs-ATCN.py
```

For a short smoke test, enable:

```python
DEBUG = True
```

Debug mode reduces the GAN and ATCN training epochs.

## Main Configuration

The principal experiment settings are defined near the top of `GANs-ATCN.py`:

```python
SEQ_LEN = 50
RUL_CAP = 125

NOISE_DIM = 32
GAN_EPOCHS = 40
GAN_BATCH = 64

ATCN_DMODEL = 64
ATCN_FILTERS = 64
ATCN_EPOCHS = 30
ATCN_BATCH = 256

LOW_DATA_FRACTION_UNITS = 0.3
VAL_FRACTION_UNITS = 0.2
```

All experiments use random seed `42` by default.

## Evaluation

The script evaluates both configurations on the real test set:

- `ATCN_real_small`: ATCN trained using the reduced real training set
- `ATCN_real_plus_syn`: ATCN trained using reduced real data plus cGAN-generated sequences

Reported metrics:

- Root Mean Square Error (RMSE)
- NASA asymmetric score

The NASA score penalizes late RUL predictions more strongly than early predictions.

## Outputs

Results are written to:

```text
results_gan_atcn/FD00X/
```

The generated artifacts include:

- trained generator and discriminator models;
- ATCN checkpoints for real-only and augmented training;
- prediction-versus-ground-truth plots;
- scatter and residual plots;
- real-versus-synthetic sequence comparisons;
- a CSV file summarizing RMSE and NASA score.

## Findings Reported in the Associated Study

The accompanying study reports that augmentation is not universally beneficial. Its value depends on whether limited training data create structural coverage gaps across health states and operating conditions.

Key reported findings include:

- 58.73% reduction in NASA Score on FD004 with cGAN augmentation;
- 93.3% reduction in NASA Score for ATCN versus the LSTM baseline on FD004;
- 47.7% reduction in Health Index forecasting RMSE in the confidential ST07 SCARA industrial case study;
- ST07 augmented-model coefficient of determination: `R² = 0.9698`.

The current public script reproduces the C-MAPSS GAN-ATCN benchmark workflow. The confidential ST07 dataset and industrial processing pipeline are not distributed.

## Reproducibility Notes

- Data splitting is performed at unit level.
- The scaler is fitted only on the reduced training split.
- Validation units are separated from the selected low-data training units.
- Test evaluation uses only real C-MAPSS sequences.
- The number of generated sequences equals the number of limited real training sequences.
- All primary random generators are seeded with `42`.

Results may vary across TensorFlow versions, hardware platforms, and GPU configurations.

## Limitations

- The current cGAN uses a lightweight adversarial objective without advanced stabilization techniques.
- Synthetic-sequence fidelity is assessed mainly through downstream forecasting performance and diagnostic plots.
- The script uses a fixed low-data fraction unless manually changed.
- The current public repository does not include the ST07 industrial data.
- The code is a research prototype and has not been packaged as a production monitoring service.
- No calibrated predictive-uncertainty module is currently implemented.

## Data Availability

The NASA C-MAPSS benchmark is publicly available through the NASA Open Data Portal.

The ST07 industrial case-study data are not publicly available because of company confidentiality requirements. They contain anonymized production information and cannot be redistributed.

## Authors

- **Aicha Benmansour**
- **Wail Rezgui**
- **Leila Zemmouchi-Ghomari**

Industrial Engineering and Maintenance Department  
National Higher School of Advanced Technologies (ENSTA), Algiers, Algeria

## Citation

The associated article is currently under review. Until a formal publication record is available, the software repository may be cited as:

```bibtex
@software{benmansour2026ganatcn,
  author  = {Benmansour, Aicha and Rezgui, Wail and Zemmouchi-Ghomari, Leila},
  title   = {GAN-ATCN for Remaining Useful Life Prediction under Data Scarcity},
  year    = {2026},
  url     = {https://github.com/zerotwoo2-k/GAN-ATCN}
}
```

## License

No software license has been added to the repository yet. Until a license is specified, reuse, modification, and redistribution are not automatically granted. Contact the authors for permission.

## Acknowledgements

This work uses the NASA C-MAPSS turbofan degradation benchmark and was developed within the Industrial Engineering and Maintenance research context at ENSTA, Algiers.
