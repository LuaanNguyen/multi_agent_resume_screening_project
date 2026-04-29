# Multi-Agent Resume Screening and Skill Mining System

**CSE 572: Data Mining - Academic Project**

This project is a Python command-line pipeline for processing resumes from the
Kaggle Resume Dataset. It extracts text and skills, trains job-category
classifiers, mines skill associations, clusters resumes, and validates PDF
extraction against CSV resume text.

The repository name includes "multi-agent", but the submitted implementation is
a modular in-process Python pipeline. The main value of the project is the data
mining workflow around resume parsing, classification, association mining,
clustering, and extraction validation.

## Start Here

If you are reviewing this project for the first time, read these files in this
order:

| File | Purpose |
| --- | --- |
| [`README.md`](README.md) | Project overview, setup, workflow, latest results, and limitations |
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`docs/REAL_DATA_RESULTS.md`](docs/REAL_DATA_RESULTS.md) | Detailed metrics from the latest real Kaggle dataset run |
| [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md) | Full command reference for every supported CLI command |
| [`docs/RESUME_PROCESSOR_ARCHITECTURE.md`](docs/RESUME_PROCESSOR_ARCHITECTURE.md) | How the resume processing pipeline is organized |
| [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) | Developer-facing module and API notes |
| [`docs/INTEGRATION_TESTS_GUIDE.md`](docs/INTEGRATION_TESTS_GUIDE.md) | Test coverage and integration test workflow |

## What The Project Does

The system takes resumes from two sources:

- A CSV file containing pre-extracted resume text
- PDF files organized by job category

It then:

1. Extracts text from resumes.
2. Splits resume text into sections such as skills, experience, education, and
   projects.
3. Extracts and normalizes skills.
4. Trains and evaluates job-category classifiers.
5. Mines frequent skill co-occurrence patterns.
6. Clusters resumes based on extracted features.
7. Compares PDF extraction output against CSV text as a validation check.

## Dataset

Dataset: Kaggle Resume Dataset by Snehaan Bhawal

Dataset link:
<https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset>

The latest local run used:

- `2484` CSV resumes
- `2484` PDF files in the archive
- `24` job categories
- `2483` successfully compared CSV/PDF resume pairs during validation

One PDF in the local archive could not be read by the extraction libraries and
was skipped during validation. The pipeline handles this case and still
completes.

## What Is Tracked In Git

The GitHub submission tracks source code, tests, configuration, and
documentation.

The following are intentionally not tracked because they are large or generated:

- `archive/` dataset files
- `output/` generated JSON reports, structured resumes, and models
- `.venv/` virtual environment
- Python caches and test caches

Run the setup and pipeline commands below to recreate the local dataset and
outputs.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip
- Kaggle access for downloading the dataset

### Setup

```bash
git clone https://github.com/Karthikp0152/multi_agent_resume_screening_project.git
cd multi_agent_resume_screening_project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python setup_dataset.py
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

`setup_dataset.py` downloads and organizes the dataset into:

- `archive/Resume/Resume.csv`
- `archive/data/data/<CATEGORY>/*.pdf`

## Main Workflow

Run commands from the repository root.

### 1. Process CSV resumes

```bash
python main.py --output-dir output process-csv --csv-file archive/Resume/Resume.csv
```

Writes structured resume JSON files to `output/csv_structured/`.

### 2. Process PDF resumes

```bash
python main.py --output-dir output process-pdf --pdf-dir archive/data/data
```

Writes structured PDF-derived resume JSON files to `output/pdf_structured/`.

### 3. Train classifiers

```bash
python main.py --output-dir output train --csv-file archive/Resume/Resume.csv
```

Trains:

- Baseline model: raw-text TF-IDF + Logistic Regression
- Proposed model: hybrid raw-text TF-IDF + normalized-skill features +
  Logistic Regression

Model artifacts are written to `output/models/`.

### 4. Evaluate classifiers

```bash
python main.py --output-dir output evaluate --csv-file archive/Resume/Resume.csv
```

Writes `output/reports/evaluation_report.json`.

### 5. Mine skill associations

```bash
python main.py --output-dir output mine --csv-file archive/Resume/Resume.csv
```

Writes `output/reports/association_rules.json`.

The mining step removes obvious resume header/contact/location artifacts before
running Apriori.

### 6. Cluster resumes

```bash
python main.py --output-dir output cluster --source csv --csv-file archive/Resume/Resume.csv
python main.py --output-dir output cluster --source pdf --pdf-dir archive/data/data
```

Writes:

- `output/reports/cluster_report.json`
- `output/reports/cluster_assignments.json`

The second cluster command overwrites the same report paths with the PDF-source
result.

### 7. Validate PDF extraction against CSV text

```bash
python main.py --output-dir output validate --csv-file archive/Resume/Resume.csv --pdf-dir archive/data/data
```

Writes `output/reports/validation_report.json`.

## Latest Real-Data Results

These are the latest verified metrics from the current checked-in code on the
Kaggle dataset. See [`docs/REAL_DATA_RESULTS.md`](docs/REAL_DATA_RESULTS.md) for
the longer breakdown.

### Classification

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Baseline: TF-IDF + Logistic Regression | 0.6258 | 0.5468 |
| Proposed: Hybrid Text + Skill Logistic Regression | 0.4165 | 0.3685 |

Finding:

- The baseline classifier outperformed the proposed hybrid classifier in the
  latest real-data run.
- The proposed model is still useful as an experiment, but the project should
  not claim it improves over the baseline.

### Association Mining

At the configured thresholds:

- `min_support`: `0.1`
- `min_confidence`: `0.5`
- Rules found: `0`

Finding:

- The cleaned transactions produced frequent itemsets, but no rules met the
  configured confidence threshold on the full dataset.

### Clustering

CSV source:

- Samples clustered: `2484`
- Actual clusters: `10`
- Silhouette score: `0.1024`
- Cluster sizes: `158`, `1`, `461`, `372`, `1`, `1`, `2`, `1`, `1`, `1486`

PDF source:

- Samples clustered: `2483`
- Actual clusters: `10`
- Silhouette score: `0.1620`
- Cluster sizes: `1`, `442`, `1`, `246`, `1`, `1`, `1`, `1788`, `1`, `1`

Finding:

- Clustering works as an exploratory feature, but the current feature space
  creates one dominant cluster and several singleton or near-singleton clusters.

### Cross-Source Validation

CSV/PDF validation results:

- Samples compared: `2483`
- Text similarity: `0.9979 +/- 0.0172`
- Skill overlap: `0.7575 +/- 0.1633`
- Extraction accuracy: `0.8777`

Finding:

- PDF text extraction closely matches the CSV text source.
- Skill extraction is less stable than raw text extraction, which is reflected
  in the lower skill-overlap score.

## Project Structure

```text
multi_agent_resume_screening_project/
|-- config/
|   |-- config.yaml
|   |-- job_categories.json
|   `-- skill_aliases.json
|-- src/
|   |-- association_miner.py
|   |-- classifier.py
|   |-- clustering_engine.py
|   |-- evaluation_module.py
|   |-- feature_generator.py
|   |-- models.py
|   |-- resume_processor.py
|   |-- scoring_engine.py
|   |-- section_parser.py
|   |-- skill_extractor.py
|   |-- skill_normalizer.py
|   `-- text_extractor.py
|-- tests/
|-- main.py
|-- setup_dataset.py
|-- requirements.txt
|-- README.md
`-- docs/
    |-- README.md
    |-- REAL_DATA_RESULTS.md
    |-- CLI_REFERENCE.md
    |-- RESUME_PROCESSOR_ARCHITECTURE.md
    |-- API_DOCUMENTATION.md
    `-- INTEGRATION_TESTS_GUIDE.md
```

Generated local folders such as `archive/` and `output/` are ignored by Git and
are recreated by the setup and pipeline commands.

## Technology Stack

- Python 3.9+
- spaCy for NLP preprocessing
- pdfplumber and pypdf for PDF text extraction
- RapidFuzz for fuzzy skill normalization
- pandas and numpy for data handling
- scikit-learn for TF-IDF, classification, and clustering
- mlxtend for Apriori association mining
- PyYAML for configuration loading

## Testing

Run all tests:

```bash
pytest
```

Run the main targeted checks:

```bash
python -m py_compile main.py src/*.py tests/*.py setup_dataset.py
pytest tests/test_main_integration.py tests/test_evaluation_module.py tests/test_association_miner.py -q
```

The most recent full local verification passed:

- `518` tests passed
- `5` warnings

## Current Scope And Limitations

- The implementation is a local Python CLI pipeline, not a deployed web
  application.
- The project name says "multi-agent", but the current implementation is not a
  runtime system of independent agents.
- The scoring engine exists in `src/scoring_engine.py`, but normal CLI
  processing currently saves structured resumes with `scores: null`.
- The proposed hybrid classifier does not outperform the baseline on the latest
  real-data run.
- Association mining returns no rules at the default confidence threshold.
- Clustering output is exploratory and currently imbalanced.

## Contributors

**CSE 572 Group 1**

- Luan Nguyen (ltnguy58@asu.edu)
- Karthik Ponugoti (kponugo2@asu.edu)
- Krish Naik (knaik13@asu.edu)
- Kiran Kamalakar (kkamala1@asu.edu)
- Sai Rithwik Reddy Chirra (schirra7@asu.edu)

## License

This is an academic project for CSE 572: Data Mining at Arizona State
University.

## Acknowledgments

- Dataset: Kaggle Resume Dataset by Snehaan Bhawal
- Course: CSE 572 - Data Mining, Arizona State University
- Research context: semantic resume matching, skill extraction, association
  mining, clustering, and fairness analysis in automated resume screening
