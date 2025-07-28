# Feature‑Level Insights into Artificial Text Detection with Sparse Autoencoders

[![ACL 2025 Findings](https://img.shields.io/badge/ACL%202025-Findings-blue)](https://aclanthology.org/2025.findings-acl.1321)
![Python >= 3.10](https://img.shields.io/badge/python-3.10%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

> **Paper:** *Feature‑Level Insights into Artificial Text Detection with Sparse Autoencoders* \
> **Venue:** Findings of ACL 2025 \
> **Authors:** Kristian Kuznetsov, Laida Kushnareva, Polina Druzhinina, Anton Razzhigaev, Anastasia Voznyuk, Irina Piontkovskaya, Evgeny Burnaev, Serguei Barannikov \
> **Link:** https://aclanthology.org/2025.findings-acl.1321/ \
> **Abstract (short):**
> “We enhance interpretability in Artificial‑Text Detection (ATD) by extracting **sparse, human‑interpretable features** from Gemma‑2‑2B with Sparse Autoencoders (SAEs) and show that a tiny fraction of those features already outperforms strong activation‑based baselines.” 

---

## Table of Contents

1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [Running the pipeline](#running-the-pipeline)

   * [1 / Extract SAE features](#1--extract-sae-features)
   * [2 / Train or evaluate XGBoost detectors](#2--train-or-evaluate-xgboost-detectors)
   * [3 / Steer the language model](#3--steer-the-language-model)
4. [BibTeX](#bibtex)
5. [Contact](#contact)

---

## Quick start

```bash
# clone & enter
git clone https://github.com/<your‑org>/SAE_ATD.git
cd SAE_ATD

# create env (conda or venv)
conda create -n sae_atd python=3.10 pytorch cudatoolkit -c pytorch -y
conda activate sae_atd

# install python deps
pip install -r requirements.txt
```

---

## Running the pipeline

Below we show the minimal commands.

### 1 / Extract SAE features

Run **once for *each* split** you want:

```bash
python scripts/run_sae_features.py \
  --dataset train        # choose from: train | dev | devtest | test
```

Outputs: `features/gemma-2-2b-res-16k-{SPLIT}.h5`

### 2 / Train or evaluate XGBoost detectors

Run **once for each feature type**:

```bash
python scripts/run_xgboost.py \
  --feature_type sae_features    # or: activations
```

*Results (`*.json`) are saved under `models/` and metrics under `results/`.*

### 3 / Steer the language model


```bash
python scripts/run_steering.py \
  --output_folder steering_outputs
```

The script

1. loads the **top‑N = 10** most important SAE features per layer from your trained XGBoost models,
2. amplifies each feature with `λ·A_max` (see §3.2 of the paper),
3. generates text continuations, saving them as JSON for later inspection.

---

## BibTeX

```bibtex
@inproceedings{kuznetsov-etal-2025-feature,
    title = "Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders",
    author = "Kuznetsov, Kristian  and Kushnareva, Laida  and Razzhigaev, Anton  and Druzhinina, Polina  and Voznyuk, Anastasia  and Piontkovskaya, Irina  and Burnaev, Evgeny  and Barannikov, Serguei",
    editor = "Che, Wanxiang  and Nabende, Joyce  and Shutova, Ekaterina  and Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1321/",
    pages = "25727--25748",
    ISBN = "979-8-89176-256-5",
}
```

---

## Contact

*Main maintainer:* Kristian Kuznetsov — [kris@kuznetsov.su](mailto:kris@kuznetsov.su)
Feel free to open an issue or reach out by e‑mail for questions or pull requests.
