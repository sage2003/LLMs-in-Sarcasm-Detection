# LLMs in Sarcasm Detection? It's Elementary! (Or Is It?)

> **Research Paper:** *LLMs in Sarcasm Detection? It's Elementary! (Or Is It?)*  
> A mechanistic interpretability study exposing why Large Language Models fail at detecting sarcasm in natural human speech.

---

## Overview

This repository contains the full experimental pipeline for the paper. The paper investigates a striking **generalization gap**: LLMs achieve near-perfect accuracy on AI-generated (synthetic) sarcasm benchmarks but collapse toward random guessing on organic, human-authored speech (Reddit posts, news headlines).

Through a multi-phase analysis — spanning behavioral evaluation, representation geometry, distributional entropy, and causal interventions — the paper demonstrates that LLMs do **not** understand the pragmatic incongruity of sarcasm. Instead, they exploit low-entropy statistical artifacts, essentially classifying the **generator** of the text rather than its **linguistic intent** (the *Clever Hans effect*).

### Key Findings

- **Synthetic Mirage:** LLMs achieve ~95–99% accuracy on synthetic sarcasm but drop to ~56–66% on human speech.
- **Hidden State Geometry:** Synthetic sarcasm forms linearly separable clusters ("clean islands") from the very first transformer layer; human sarcasm remains entangled through all layers.
- **Shortcut Learning via Min-K%++:** Models use statistical familiarity (low token-probability variance) as a proxy for sarcasm — treating predictable AI text as "more sarcastic."
- **Semantic Blindness:** Word occlusion and LIME show models remain confident in sarcasm predictions even when the words that *make* a sentence sarcastic are fully masked.

---

## Repository Structure

```
.
├── README.md
├── Datasets/                         # All input datasets (see setup below)
│   ├── SARC-V1/                      # Reddit SARC V1 (organic)
│   │   ├── SARC-V1.csv
│   │   ├── SARC-V1-shuffled.csv      # Shuffled version (for few-shot sampling)
│   │   └── SARC-V1-ablation_fixed.csv # 100-row fixed subset for ablation
│   ├── SARC-V2/                      # Reddit SARC V2 subsets (organic)
│   │   ├── HYP-sarc-notsarc.csv      # Hyperbole
│   │   ├── GEN-sarc-notsarc.csv      # General
│   │   └── RQ-sarc-notsarc.csv       # Rhetorical Questions
│   ├── News_Headlines/               # News Headlines Dataset (organic)
│   │   └── Sarcasm_Headlines_Dataset.csv
│   ├── Synthetic/                    # LLM-generated sarcasm datasets
│   │   ├── Llama-3.3-70b-versatile/
│   │   │   └── sarcasm_dataset_llama3.3-70b.csv
│   │   └── kimi-k2-instruct-0905/
│   │       └── sarcasm_dataset_kimi-k2-instruct-0905.csv
│   ├── Intermediate/                 # Hybrid few-shot anchored dataset
│   │   └── anchored_fewshot_sarc_llama70b.csv
│   └── interpretability_sentences.csv # Fixed sentences for Phases 6, 7, 10
│
└── Codes/                            # All analysis scripts
    ├── dataset_generator.py                          # Generate synthetic dataset (Groq API)
    ├── dataset_generator_fewshot.py                  # Generate few-shot anchored dataset
    ├── dataset_regeneration.py                       # Alternate synthetic generation script
    ├── dataset_statistics.py                         # Dataset statistics table (Table 1 in paper)
    ├── zero_shot_classification_phase1.py            # Phase 1: Zero-shot classification metrics
    ├── zero_shot_classification_plot_phase1.py       # Phase 1: Plot classification results
    ├── embedding_geometry_phase2.py                  # Phase 2: Hidden state geometry (t-SNE / UMAP)
    ├── embedding_geometry_plot_phase2.py             # Phase 2: Plot embedding visualizations
    ├── perplexity_analysis_phase4.py                 # Phase 4: Sentence perplexity (100-run Monte-Carlo)
    ├── perplexity_analysis_plot_phase4.py            # Phase 4: Plot perplexity distributions
    ├── minkpp_analysis_phase5.py                     # Phase 5: Min-K%++ membership inference
    ├── minkpp_analysis_plot_phase5.py                # Phase 5: Plot Min-K%++ distributions
    ├── interpretability_sentence_selector_phase5.5.py # Phase 5.5: Generate interpretability_sentences.csv
    ├── lime_explanations_phase6.py                   # Phase 6: LIME local explanations
    ├── attention_visualization_phase7.py             # Phase 7: Attention head visualization (BertViz)
    ├── ablation_dataset_gen_phase8.py                # Phase 8: Generate SARC-V1 ablation subsets
    ├── layer_ablation_phase8.py                      # Phase 8: Layer & chunk-wise block ablation
    ├── layer_ablation_plot_phase8.py                 # Phase 8: Plot ablation accuracy drops
    ├── token_occlusion_phase10.py                    # Phase 10: Word-level token occlusion
    ├── Outputs/                                      # Generated plots and CSVs (gitignored)
    └── Final_Plots/                                  # Publication-ready figures
```

---

## Experimental Phases

The pipeline is organized into numbered phases, each corresponding to a section of the paper:

| Phase | Script | Description |
|-------|--------|-------------|
| **Dataset Setup** | `dataset_generator.py` | Generates synthetic sarcasm data via Groq API (Llama-3.3-70B) |
| **Dataset Setup** | `dataset_generator_fewshot.py` | Generates the hybrid "Natural-Generated" dataset using few-shot prompting anchored to SARC-V1 examples |
| **Table 1** | `dataset_statistics.py` | Computes size, label distribution, avg. length, and type-token ratio for all 8 datasets |
| **Phase 1** — Zero-Shot Classification | `zero_shot_classification_phase1.py` | Zero-shot binary sarcasm classification; reports Accuracy, Precision, Recall, F1, and confusion matrices |
| **Phase 2** — Embedding Geometry | `embedding_geometry_phase2.py` | Extracts mean-pooled hidden states from first/middle/last layers; projects to 2D/3D via t-SNE and UMAP |
| **Phase 4** — Perplexity Analysis | `perplexity_analysis_phase4.py` | Computes per-sentence perplexity across all datasets; 100-run Monte-Carlo averaging for stability |
| **Phase 5** — Min-K%++ Analysis | `minkpp_analysis_phase5.py` | Computes Min-K%++ scores (normalized token log-probability variance) to measure statistical familiarity |
| **Phase 5.5** — Sentence Selection | `interpretability_sentence_selector_phase5.5.py` | Selects representative sentences per dataset for interpretability analysis |
| **Phase 6** — LIME Explanations | `lime_explanations_phase6.py` | Runs LIME perturbation-based local explanations on selected sentences |
| **Phase 7** — Attention Visualization | `attention_visualization_phase7.py` | Generates BertViz attention head visualizations |
| **Phase 8** — Layer Ablation | `layer_ablation_phase8.py` | Causal layer ablation: replaces transformer blocks with identity modules, measures accuracy drop |
| **Phase 10** — Token Occlusion | `token_occlusion_phase10.py` | Word-level occlusion: masks individual tokens and tracks change in sarcasm probability |

> **Note:** Phases 3 and 9 were exploratory experiments conducted during development that were not included in the final paper — hence the numbering gap.

---

## Models Used

All phases evaluate four instruction-tuned LLMs loaded in 4-bit NF4 quantization via `bitsandbytes`:

| Model | HuggingFace ID |
|-------|----------------|
| LLaMA-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` |
| Phi-3-Medium-4K-Instruct | `microsoft/Phi-3-medium-4k-instruct` |
| Gemma-2-9B-IT | `google/gemma-2-9b-it` |


---

## Datasets

The organic datasets must be downloaded manually from their official sources and placed in the correct subdirectories. The synthetic and hybrid datasets can be reproduced using the provided generator scripts.

### Organic Datasets (download required)

#### SARC V1 — Reddit Self-Annotated Corpus
> Khodak et al. (2018) — *A Large Self-Annotated Corpus for Sarcasm*

| Resource | Link |
|----------|------|
| Download (Kaggle) | https://www.kaggle.com/datasets/danofer/sarcasm |
| Official GitHub | https://github.com/NLPrinceton/SARC |
| Paper (arXiv) | https://arxiv.org/abs/1704.05579 |

Download the dataset and place the CSV files as:
```
Datasets/SARC-V1/SARC-V1.csv
Datasets/SARC-V1/SARC-V1-shuffled.csv   # a shuffled copy of SARC-V1.csv
```
> `SARC-V1-shuffled.csv` is just `SARC-V1.csv` randomly shuffled — run `df.sample(frac=1, random_state=42).reset_index(drop=True)` and save.

#### SARC V2 — Sarcasm Corpus V2 (Oraby et al.)
> Oraby et al. (2016) — *Creating and Characterizing a Diverse Corpus of Sarcasm in Dialogue*

| Resource | Link |
|----------|------|
| Download (Kaggle) | https://www.kaggle.com/datasets/coldn00ldes/sarcasm-corpus-v2oraby-et-al |
| Dataset Homepage | https://nlds.soe.ucsc.edu/sarcasm2 |
| Paper (ACL Anthology) | https://aclanthology.org/W16-3604/ |
| Paper (arXiv) | https://arxiv.org/abs/1709.05404 |

Download the dataset and place the three subset CSV files as:
```
Datasets/SARC-V2/HYP-sarc-notsarc.csv   # Hyperbole subset
Datasets/SARC-V2/GEN-sarc-notsarc.csv   # General subset
Datasets/SARC-V2/RQ-sarc-notsarc.csv    # Rhetorical Questions subset
```

#### News Headlines Sarcasm Dataset
> Misra & Arora (2019) — *Sarcasm Detection using Hybrid Neural Network*

| Resource | Link |
|----------|------|
| Download (Kaggle) | https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection |
| Official GitHub | https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection |
| Paper (arXiv) | https://arxiv.org/abs/1908.07414 |

Download and place as:
```
Datasets/News_Headlines/Sarcasm_Headlines_Dataset.csv
```

---

### Synthetic Datasets (reproducible via scripts)

These are generated using the Groq API and can be fully reproduced. If you want to skip generation, contact the authors for access to the pre-generated files.

| Name | Generator Script | Output Path |
|------|-----------------|-------------|
| Llama70B Synthetic | `dataset_generator.py` | `Datasets/Synthetic/Llama-3.3-70b-versatile/sarcasm_dataset_llama3.3-70b.csv` |
| KimiK2 Synthetic | *(modify `dataset_generator.py` to use `kimi-k2-instruct-0905`)* | `Datasets/Synthetic/kimi-k2-instruct-0905/sarcasm_dataset_kimi-k2-instruct-0905.csv` |
| Natural-Generated | `dataset_generator_fewshot.py` | `Datasets/Intermediate/anchored_fewshot_sarc_llama70b.csv` |

> **Note:** Requires a [Groq API key](https://console.groq.com) set via `os.environ["GROQ_API_KEY"]` in the generator scripts.

---

## Setup & Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (minimum 16GB VRAM for 4-bit quantized 14B models; 8–10GB sufficient for 8B/9B models)

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers bitsandbytes accelerate
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
pip install umap-learn
pip install lime
pip install bertviz
pip install groq          # only needed for dataset generation
pip install scipy
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### HuggingFace Login (for gated models)

```bash
huggingface-cli login
```

### Groq API Key (for dataset generation only)

Set your key in `dataset_generator.py` and `dataset_generator_fewshot.py`:

```python
os.environ["GROQ_API_KEY"] = "your_key_here"
```

---

## Reproducing the Experiments

All scripts are designed to be run from the **`Codes/`** directory.

```bash
cd Codes/
```

### Step 0: Prepare Datasets

Download the SARC V1 and V2 datasets and the News Headlines dataset and place them in their respective `Datasets/` subdirectories as shown in the structure above.

To regenerate the synthetic datasets (requires Groq API key):

```bash
python dataset_generator.py            # Llama-70B synthetic set
python dataset_generator_fewshot.py    # Natural-Generated hybrid set
```

### Step 1: Dataset Statistics (Table 1)

```bash
python dataset_statistics.py
```

### Step 2: Zero-Shot Classification (Phase 1)

```bash
python zero_shot_classification_phase1.py
python zero_shot_classification_plot_phase1.py
```

### Step 3: Embedding Geometry (Phase 2)

```bash
python embedding_geometry_phase2.py
python embedding_geometry_plot_phase2.py
```

### Step 4: Perplexity Analysis (Phase 4)

> ⚠️ This runs 100 Monte-Carlo subsampling rounds per model across 8 datasets. Expect **several hours** per model on a single GPU.

```bash
python perplexity_analysis_phase4.py
python perplexity_analysis_plot_phase4.py
```

### Step 5: Min-K%++ Analysis (Phase 5)

```bash
python minkpp_analysis_phase5.py
python minkpp_analysis_plot_phase5.py
```

### Step 5.5: Generate Interpretability Sentences

Run this **before** Phases 6, 7, and 10 — it creates `interpretability_sentences.csv` used by all three:

```bash
python interpretability_sentence_selector_phase5.5.py
```

### Step 6: LIME Explanations (Phase 6)

```bash
python lime_explanations_phase6.py
```

### Step 7: Attention Visualization (Phase 7)

```bash
python attention_visualization_phase7.py
```

### Step 8: Layer Ablation (Phase 8)

First, generate the fixed 100-row SARC-V1 ablation subset (run once):

```bash
python ablation_dataset_gen_phase8.py
```

Then run the main ablation:

```bash
python layer_ablation_phase8.py
python layer_ablation_plot_phase8.py
```

### Step 10: Token Occlusion (Phase 10)

```bash
python token_occlusion_phase10.py
```

---

## Output Structure

All outputs (CSVs, PNGs, PDFs) are written to directories created automatically at runtime:

```
Codes/
├── Outputs/
│   ├── Phase-2/             # Embedding scatter plots (t-SNE / UMAP)
│   ├── Phase-4/             # Perplexity histograms and KDE plots
│   └── ...                  # Min-K++, ablation outputs
├── CSV/                     # Intermediate averaged data (for plotting scripts)
│   ├── perplexity_hist_avg_<model>.csv
│   ├── logppl_kde_avg_<model>.csv
│   ├── minkpp_hist_avg_<model>.csv
│   └── ...
├── lime_outputs/            # LIME HTML explanation files
├── ablation_outputs/        # Layer/chunk ablation CSVs and plots
└── second_phase10_token_occlusion/  # Word occlusion heatmaps
```

---

## Reproducibility Notes

- **Random seeds** are fixed at `42` throughout all scripts (`RANDOM_SEED = 42`).
- **Monte-Carlo averaging** (Phases 4 & 5) runs 100 rounds of 1,100-point subsampling per dataset to ensure stable distribution estimates.
- **4-bit quantization** (`NF4`, double quant, `bfloat16` compute dtype) is used by default for all models. Set `USE_4BIT = False` in any script to disable it (requires significantly more VRAM).
- **Batch sizes** are tuned conservatively; lower `BATCH_SIZE` in any script if you encounter OOM errors.

---

## Citation

If you use this code or build on our findings, please cite:

```bibtex
@inproceedings{llms-sarcasm-detection-2024,
  title     = {LLMs in Sarcasm Detection? It's Elementary! (Or Is It?)},
  booktitle = {Proceedings of ...},
  year      = {2024}
}
```

---

## License

This repository is released for research purposes. See `LICENSE` for details.
