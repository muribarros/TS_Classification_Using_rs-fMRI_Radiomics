# Pediatric Tourette Syndrome Classification Using rs-fMRI Radiomics of the CSTC Circuit: A Genetic Algorithm-Guided Approach

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-blue)](#)
[![FAPESP](https://img.shields.io/badge/Funded_by-FAPESP_2025%2F00214--8-yellow)](#)
[![Made with Love](https://img.shields.io/badge/Made%20with❤️-by%20Murilo%20C.%20de%20Barros-red)](#)

**Murilo C. de Barros · Kaue T. N. Duarte · Chia-Jui Hsu · Wang-Tso Lee · Marco A. G. de Carvalho**

---

## Overview

This repository contains the **scripts** and **resources** used in our paper:

> **"Pediatric Tourette Syndrome Classification Using rs-fMRI Radiomics of the CSTC Circuit: A Genetic Algorithm-Guided Approach"**

Our work proposes an integrated computational framework for automated classification of **Tourette's Syndrome (TS)** in pediatric populations, focusing on the **Cortico-Striatal-Thalamic-Cortical (CSTC)** circuit. The pipeline combines:

- 🧠 **Spectral Analysis** — BOLD time-series converted to static 3D ALFF maps via Fast Fourier Transform (FFT)
- 🔬 **Functional Radiomics** — 3D Grey-Level Co-occurrence Matrix (GLCM) texture features extracted per CSTC region
- 🧬 **Evolutionary Feature Selection** — Block-constrained Genetic Algorithm (GA) operating at the anatomical region level
- 🤖 **Supervised Classification** — Naive Bayes, SVM (Linear/RBF), and Multilayer Perceptron architectures

The proposed framework achieved a peak diagnostic accuracy of **78.00%** using a 3-layer MLP with only **4 anatomical structures** (96 radiomic features), consistently highlighting left-hemisphere structures as the most discriminative biomarkers.

---

## Directory Structure

```
.
├── scripts/
│   ├── preprocessing/
│   │   └── motion_correction.py       # 4D rigid-body realignment via FLIRT (FSL)
│   ├── alff/
│   │   └── compute_alff.py            # FFT-based ALFF map generation
│   ├── radiomics/
│   │   └── glcm_extraction.py         # 3D GLCM texture feature extraction per ROI
│   ├── ga/
│   │   └── genetic_algorithm.py       # Block-constrained Genetic Algorithm for ROI selection
│   └── classification/
│       └── train_classifiers.py       # Naive Bayes, SVM, MLP training with 5-Fold CV
├── data/
│   └── README_data.md                 # Instructions for dataset access (NTU Hospital)
├── results/
│   └── selected_rois_per_model.csv    # GA-selected ROIs and accuracy per classifier
└── README.md
```

---

## 🧩 Pipeline Overview

The framework is structured into **6 sequential phases**:

| Step | Phase | Description |
|------|-------|-------------|
| 1 | **Dataset Organization** | rs-fMRI data split into TS (n=34) and Healthy Controls (n=34) |
| 2 | **4D Spatial Preprocessing** | Intra-functional rigid-body motion correction (FLIRT, 6 dof) |
| 3 | **Fourier Spectral Analysis** | FFT → ALFF maps (0.01–0.08 Hz neurophysiological band) |
| 4 | **Feature Extraction (GLCM)** | 24 3D radiomic descriptors per CSTC region (d=1, 4 orientations) |
| 5 | **Genetic Algorithm Selection** | Block-based GA over 17 ROIs × 24 features (100 generations, pop=50) |
| 6 | **Classification** | QuantileTransformer normalization + supervised ML classifiers |

---

## 🧠 CSTC Regions of Interest (17 ROIs)

The genetic algorithm operates over the following anatomical structures of the CSTC circuit:

| Region | Hemisphere |
|--------|------------|
| Caudate Nucleus | Left / Right |
| Thalamus Proper | Left / Right |
| Putamen | Left / Right |
| Accumbens Area | Left / Right |
| Ventral Diencephalon (Ventral DC) | Left / Right |
| Medial Orbitofrontal Cortex (mOFC) | Left / Right |
| Superior Parietal Cortex | Left / Right |
| Inferior Parietal Cortex | Left / Right |
| Lateral Orbitofrontal Cortex (LOFC) | Left only |

---

## 📊 Results

Classification performance across all models using 5-Fold Stratified Cross-Validation:

| Model | Selected ROIs | # ROIs | # Features | Accuracy |
|-------|---------------|--------|------------|----------|
| Naive Bayes | Caudate (L,R), Lat. OFC (L), Sup. Parietal (L,R) | 5 | 120 | 71.21% |
| SVM (Linear) | Accumbens (L), Lat. OFC (L), Thalamus (L,R), Ventral DC (R) | 5 | 120 | 72.73% |
| SVM (RBF) | Accumbens (L), Caudate (L,R), Lat. OFC (L) | 4 | 96 | 72.73% |
| **MLP (3 layers)** | **Accumbens (L), Caudate (L,R), Thalamus (L)** | **4** | **96** | **78.00%** |
| MLP (4 layers) | Accumbens (L), Lat. OFC (L), Med. OFC (R), Caudate (R), Ventral DC (L,R) | 6 | 144 | 71.21% |
| MLP (5 layers) | Accumbens (L), Med. OFC (L), Putamen (L), Ventral DC (R) | 4 | 96 | 72.73% |

**Most selected structures across all models:**

| Structure | Times Selected (out of 6) |
|-----------|--------------------------|
| Left Nucleus Accumbens | 5 |
| Left Caudate Nucleus | 4 |
| Left Thalamus Proper | 4 |

---

## 🚀 Usage

### Dependencies

```bash
pip install numpy scipy scikit-learn nibabel nilearn pandas
```

FSL must be installed and available in your PATH for the preprocessing step (`FLIRT`).

### Running the Pipeline

```bash
# Step 1 – Motion correction
python scripts/preprocessing/motion_correction.py --input /path/to/4D_fmri.nii.gz --output /path/to/output/

# Step 2 – Compute ALFF maps
python scripts/alff/compute_alff.py --input /path/to/realigned.nii.gz --tr 2.0 --output /path/to/alff_map.nii.gz

# Step 3 – Extract GLCM features
python scripts/radiomics/glcm_extraction.py --alff /path/to/alff_map.nii.gz --atlas /path/to/cstc_atlas.nii.gz --output /path/to/features.csv

# Step 4 – Run Genetic Algorithm
python scripts/ga/genetic_algorithm.py --features /path/to/features.csv --labels /path/to/labels.csv --output /path/to/selected_rois/

# Step 5 – Train classifiers
python scripts/classification/train_classifiers.py --features /path/to/selected_features.csv --labels /path/to/labels.csv
```

---

## 📁 Dataset

The rs-fMRI dataset used in this study was obtained through an institutional partnership with the **National Taiwan University Hospital** (NTUH), Taipei, Taiwan. It comprises **68 pediatric subjects** (ages 6–14):

- **34 Tourette's Syndrome** patients
- **34 Healthy Controls**
- Acquisition: 3T Siemens TrioTim, TR=2.0s, TE=26ms, 34 axial slices, 180 volumes

> ⚠️ The dataset is not publicly distributed. Please refer to `data/README_data.md` for institutional access information.

---

## License

This project is licensed under the MIT License — you're free to use, modify, and distribute with attribution.

---

## Citation

The citation will be updated once this paper is published. Our citation template is:

```bibtex
@article{deBarros2026TSradiomics,
  title     = {Pediatric Tourette Syndrome Classification Using rs-fMRI Radiomics of the CSTC Circuit: A Genetic Algorithm-Guided Approach},
  author    = {Murilo Costa de Barros and Kau{\^e} Tartarotti Nepomuceno Duarte and Chia-Jui Hsu and Wang-Tso Lee and Marco Antonio Garcia de Carvalho},
  journal   = {Journal of LaTeX Class Files},
  year      = {2026},
  volume    = {NA},
  pages     = {NA},
  doi       = {NA}
}
```

---

## Acknowledgments

This study was financed in part by the **São Paulo Research Foundation (FAPESP)**, Brazil – Process Number **2025/00214-8**; and partially supported by the **Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES)** – Finance Code 001.

We thank the Department of Pediatric Neurology, National Taiwan University, Taipei, Taiwan, for providing the neuroimaging data used in this work.
