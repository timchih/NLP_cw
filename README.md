# NLP Coursework: PCL Detection

Binary classification for **Patronizing and Condescending Language (PCL)** on the [Don't Patronize Me](https://competitions.codalab.org/competitions/26190) (SemEval 2022 Task 4) dataset. The final model is a **RoBERTa-base** classifier with class-imbalance handling (decreasing class-weight schedule) and HTML-tag preprocessing.

## Project structure

```
NLP_cw/
├── data/
│   └── raw/                    # Place dataset files here (see Data setup)
├── models/                     # Saved model checkpoints (created by training)
├── output/                     # Predictions, metrics, confusion matrices, error analysis
├── src/
│   ├── data_exploration.ipynb  # Exploratory data analysis
│   ├── pcl_imbalance_final.ipynb  # Final training & evaluation notebook
│   └── summarize_predictions.py   # Summarise dev predictions and compare to baseline
├── dev.txt                     # Dev-set predictions (written by final notebook)
├── baseline_dev.txt            # Optional: baseline predictions for error analysis
├── requirements.txt
└── README.md
```

## Environment setup

### 1. Python

Use **Python 3.10+** (3.12 used in development). Create and activate a virtual environment:

```bash
cd /path/to/NLP_cw
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: `transformers`, `torch`, `pandas`, `scikit-learn`, `datasets`, `accelerate`, `nltk`, `matplotlib`, `seaborn`, `optuna`.

### 3. NLTK data (for data exploration)

The data exploration notebook uses NLTK. Download required data once:

```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
"
```

## Data setup

Place the Don't Patronize Me raw data under `data/raw/`. Required files:

| File | Description |
|------|-------------|
| `dontpatronizeme_pcl.tsv` | Main dataset (paragraphs and labels) |
| `train_semeval_parids-labels.csv` | Train split (par_id, label) |
| `dev_semeval_parids-labels.csv` | Dev split (par_id, label) |
| `task4_test.tsv` | Test set (optional; for final predictions) |

Labels in the main TSV use the 0–4 scale; the code maps **0,1 → 0** (no PCL) and **2,3,4 → 1** (PCL) for binary classification.

## How to run

All commands below assume the project root is the current directory and the virtual environment is activated.

### Final notebook (`pcl_imbalance_final.ipynb`)

Trains RoBERTa-base with a decreasing class-weight schedule and HTML preprocessing, evaluates on the dev set, and writes dev predictions to `dev.txt`.

- **Interactively:** open `src/pcl_imbalance_final.ipynb` in Jupyter and run all cells. The notebook resolves the project root from the current working directory (works when Jupyter is started from the project root).
- **Headless (e.g. SLURM):** use the provided script with the notebook path:

  ```bash
  NOTEBOOK=src/pcl_imbalance_final.ipynb bash run_notebook.sh
  ```

Training uses CUDA if available; checkpoints are saved under `models/roberta_pcl_imbalance_final/`.

### Data exploration (`data_exploration.ipynb`)

Exploratory analysis of the dataset: label distribution, text length, word clouds, n-grams, etc.

- The notebook uses `BASE_DIR = Path('../data/raw')`, so it expects the **current working directory to be `src`** when cells run. Easiest: start Jupyter from the project root, open `src/data_exploration.ipynb`, and in the first cell change to `BASE_DIR = Path('data/raw')` so that data is found at `data/raw` relative to the project root. Then run all cells.
- Or run from a shell with cwd set to `src`:

  ```bash
  cd src && jupyter nbconvert --to notebook --execute data_exploration.ipynb --inplace
  ```

### Summarize predictions (`summarize_predictions.py`)

Reads `dev.txt` (and optionally `baseline_dev.txt`), prints counts, confusion matrix, recall/precision/accuracy/F1, and writes:

- `output/dev_failing_cases.tsv` — FP/FN cases  
- `output/dev_confusion_matrix.png` — confusion matrix plot  
- `output/error_analysis_examples.tsv` — stratified comparison vs baseline (if `baseline_dev.txt` exists)

**`dev.txt` format:** either (1) one prediction per line (0 or 1), same order as the dev set, or (2) TSV with header `label_binary\tpred_label`.

Run from the **project root**:

```bash
python src/summarize_predictions.py
```

Ensure `dev.txt` exists (produced by the final notebook). For baseline comparison, place baseline dev predictions in `baseline_dev.txt` (one label per line, same order as dev).

## Optional: batch run with SLURM

`run_notebook.sh` is set up for a SLURM GPU job. It backs up the notebook, runs it with `jupyter nbconvert --execute`, and logs to `logs/`. Customize partition and paths as needed:

```bash
export NOTEBOOK=src/pcl_imbalance_final.ipynb
sbatch run_notebook.sh
```

## Outputs

- **`dev.txt`** — Dev-set predictions from the final model.  
- **`output/`** — Metrics, confusion matrix, failing cases, and (if baseline provided) error-analysis examples.  
- **`models/roberta_pcl_imbalance_final/`** — Model checkpoints and config from the final notebook.

## License and data

Dataset and task are from SemEval 2022 Task 4 (Don't Patronize Me). Comply with the competition’s terms when using the data.
