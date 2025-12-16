# LEARN: LLM-augmented clinical risk scores identify preclinical patients at risk for progression to systemic lupus erythematosus

## Introduction

Systemic lupus erythematosus (SLE) is a severe autoimmune condition with no cure. SLE patients often experience a pre-clinical state where they have autoimmunity, e.g., having positive anti-nuclear antibody (ANA+), and a subset of symptoms, but do not meet full diagnostic criteria. Only a portion of preclinical patients develop SLE. Identifying patients at risk for disease progression and implementing early interventions would be crucial for mitigating disease morbidity and mortality, as well as improving quality of life. 

Electronic health records (EHR) contain health histories that can be used to predict SLE progression from preclinical stages. Large language models have been increasingly used to analyze EHR data, but their utility in clinical risk prediction remains underexplored. 

We develop a novel approach, **LLM-Embedding Augmented Risk evaluatioN (LEARN)**, which augments structured EHR with LLM-embeddings. Our results show that LEARN can substantially improve prediction accuracy over methods that rely solely on embedding or billing codes. When applied to predicting SLE progression from the preclinical stage, LEARN achieves an area under the curve of 0.83, representing an improvement of up to 0.17 over models using billing codes alone and up to 0.20 over methods using embeddings alone. The improvement is consistent regardless of the machine learning models used. Using larger LLMs to generate embeddings tends to yield even better prediction accuracy. The improvement of LEARN transfers well between biobanks. Genetic risk scores complement clinical risk scores and further stratify patients with high LEARN scores. LEARN scores can also identify patients at risk for more severe SLE-related outcomes, including lupus nephritis. Together, our work provides a general framework for using LLM-embeddings to accurately predict autoimmune progression risks, which is broadly applicable to other disease areas. 

<p align="center">
  <img src="./fig/Figure1.png" width="700" />
</p>

<p align="center">
  <b>Fig.1</b> LEARN Framework: LLM-Embedding Augmented Risk evaluatioN
</p>

## About this repository

This repository provides the codebase for the LEARN framework, including:
- Scripts for generating LLM-based clinical embeddings using [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2), a 7B instruction-tuned model based on Mistral-7B
- Model definitions and training pipelines for risk prediction
- Utilities for evaluation and visualization

⚠️ **Data privacy and availability**

This repository contains **code only**. No patient-level electronic health
record (EHR) data are included or distributed due to privacy, ethical, and
regulatory constraints.

The data used in this study were obtained from **[TriNetX](https://trinetx.com/)** and are subject to
institutional review board (IRB) approval and data use agreements. Access to
these data may be granted to qualified researchers through TriNetX.

The provided code can be adapted for use with other compliant EHR datasets
with similar structure.

## ### Required input files (not included)

The LEARN pipeline expects the following input files under `data_folder`
(default: `../data/`):

- `final_data_yrCut0.txt`  
  A tab-separated cohort-level EHR table indexed by `patient_id`.  
  This file contains patient demographics, ANA+ index dates, outcome labels
  (SLE progression), and aggregated EHR features.

- `diagnosis.csv`  
  Raw diagnosis records (ICD codes) for all patients in the cohort.

- `procedure.csv`  
  Raw procedure records (ICD-9-PCS / CPT codes) for all patients in the cohort.

- `medication_drug.csv`  
  Raw medication records (RxNorm and/or NDC codes) for all patients in the cohort.

- `standardized_terminology.csv`  
  Mapping table used to convert clinical codes into human-readable text.

- `RxNorm_full_12022024/rrf/RXNSAT.RRF`  
- `RxNorm_full_12022024/rrf/RXNCONSO.RRF`  
  RxNorm reference files used to normalize medication codes.

Derived cohort-level summaries and embeddings are generated automatically and
saved under `processed_data/` and `embeddings/`.

---

### Notes on system-specific paths

Some paths in the code are **environment-dependent** and may need to be updated:

- The NV-Embed-v2 model is loaded from a local Hugging Face cache directory.
  Users should update this path or load the model directly from Hugging Face.
- The `HF_HOME` environment variable may need adjustment depending on the system.
  ```python
  local_model_path = "/path/to/huggingface/cache/models--nvidia--NV-Embed-v2/..."
  
## Implementation

### Environment

We provide a python [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment used to generate embeddings, train models, and reproduce all reported results.

** System information and important package versions:**

System: `Red Hat Enterprise Linux 8.10 (Ootpa)`;

Python version: `3.11.7`;

GPU: `NVIDIA A100-PCIE-40GB`;

`pytorch`: `2.1.2`;

`scikit-learn`: `1.2.2`.

All other dependencies will be downloaded when the provided environment is imported.

## Setup

### Clone Repository

```bash
git clone https://github.com/ya61sen/LEARN.git
cd LEARN
```

### Create and import Conda Environment

```bash
conda env create -f environment.yml
source activate learn_env
```

## Run LEARN

### Model training

`train.py` trains the LEARN risk prediction model with configurable cohort windows:
- LIS: length in system (days)
- BP: buffer period (days)
- FUP: follow-up period (days)
- --include_med_emb: optional flag to include medication embeddings

1) Train with default configuration

```bash
python src/train.py --include_med_emb 
```
Defaults:
- --LIS: 2*365 (730 days)
- --BP: 180 days
- --FUP: 5*365 (1825 days)
- medication embeddings: enabled unless `--include_med_emb` is removed

2) Train with custom time windows

Example (custom LIS/BP/FUP plus medication embeddings):
```bash
python src/train.py --LIS 365 --BP 365 --FUP 730 --include_med_emb 
```

### What happens when you run it

When executed, train.py:
1. Prints the selected configuration
2. Builds the training dataset via:
	`generate_embedding_data(LIS, BP, FUP)`
3.	Preprocesses features and labels via:
	`preprocess_data(..., include_med_emb)`
4.	Trains the predictive model via:
	`train_model(X, y, LIS, BP, FUP, include_med_emb=...)`

## Citation

If you use this code, please cite:

`LEARN: LLM-augmented clinical risk scores identify preclinical patients at risk for progression to systemic lupus erythematosus`

(Citation details to be added upon publication.)

## Contacts

**Sen Yang:**

sky5218@psu.edu | syang4@pennstatehealth.psu.edu

Department of Public Health Sciences

Penn State College of Medicine

Hershey, PA 17033
