# Multi-Agent Resume Screening & Skill Mining System

**CSE 572: Data Mining - Academic Project**

An NLP-powered resume processing pipeline that extracts, normalizes, and analyzes resume data from the Kaggle Resume Dataset.

## Project Overview

This repository currently supports a modular in-process pipeline that:
- Extracts structured information from CSV and PDF resume sources
- Parses resume sections and normalizes extracted skills
- Trains and evaluates job-category classifiers
- Mines association rules from normalized skill data
- Validates PDF extraction quality against CSV ground truth

The latest real-data run used the Kaggle Resume Dataset with:
- 2,484 CSV resumes
- 24 job categories
- 2,483 successfully compared CSV/PDF resumes in the validation run

## Key Features

### Pipeline Components
- **Text Extraction**: Converts PDF resumes to text using `pdfplumber` with fallback extraction support
- **Section Parsing**: Divides resumes into logical sections (Skills, Experience, Education, Projects)
- **Skill Extraction**: Uses spaCy NLP to identify explicit and implicit skills
- **Skill Normalization**: Standardizes skill variations using fuzzy matching (`RapidFuzz`)
- **Scoring Engine**: ATS and semantic scoring module exists in `src/scoring_engine.py`, but normal CLI processing currently saves structured resumes with `scores: null`

### CLI-Supported Analysis
- **Classification**: Trains a TF-IDF + Logistic Regression baseline and a skill-feature + Random Forest proposed model
- **Association Mining**: Discovers co-occurring normalized tokens using Apriori
- **Fairness Analysis**: Evaluates per-category F1 for the proposed classifier
- **Cross-Source Validation**: Compares PDF-extracted text/skills against CSV ground truth

### Implemented Module Not Exposed by CLI
- **Clustering**: `src/clustering_engine.py` exists and is tested, but there is currently no `cluster` command in `main.py`

### Dual Data Source Support
- **CSV Processing**: Fast processing using pre-extracted text from `archive/Resume/Resume.csv`
- **PDF Processing**: Full extraction pipeline validation from `archive/data/data/` organized by job categories
- **Cross-Validation**: Compares PDF extraction accuracy against CSV ground truth

## Technology Stack

**Core Language**: Python 3.9+

**NLP & ML Libraries**:
- `spaCy` 3.x - NER for skill extraction
- `sentence-transformers` - Semantic embeddings (all-MiniLM-L6-v2)
- `scikit-learn` - Classification, clustering, TF-IDF
- `mlxtend` - Apriori algorithm for association mining

**Text Processing**:
- `pdfplumber` - PDF text extraction
- `RapidFuzz` - Fuzzy string matching

**Data Processing**:
- `pandas`, `numpy` - Data manipulation
- `PyYAML` - Configuration management

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Karthikp0152/multi_agent_resume_screening_project.git
cd multi_agent_resume_screening_project
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

5. **Download and organize the Kaggle dataset**
```bash
python setup_dataset.py
```

This populates:
- `archive/Resume/Resume.csv`
- `archive/data/data/<CATEGORY>/*.pdf`

## Usage

### Command-Line Interface

The CLI commands below match the workflow that currently runs successfully in this repo:

#### 1. Process CSV Resumes
```bash
python main.py --output-dir output process-csv --csv-file archive/Resume/Resume.csv
```

#### 2. Process PDF Resumes
```bash
python main.py --output-dir output process-pdf --pdf-dir archive/data/data
```

#### 3. Train ML Models
```bash
python main.py --output-dir output train --csv-file archive/Resume/Resume.csv
```

#### 4. Evaluate Models
```bash
python main.py --output-dir output evaluate --csv-file archive/Resume/Resume.csv
```

#### 5. Mine Skill Associations
```bash
python main.py --output-dir output mine --csv-file archive/Resume/Resume.csv
```

#### 6. Cross-Source Validation
```bash
python main.py --output-dir output validate --csv-file archive/Resume/Resume.csv --pdf-dir archive/data/data
```

### Configuration

Edit `config/config.yaml` to customize:
- PDF extraction method (pdfplumber/pypdf)
- NLP model selection
- Fuzzy matching threshold
- ML hyperparameters (test split, clustering config, min_support, min_confidence)

## Project Structure

```
multi_agent_resume_screening_project/
├── src/                          # Source code
│   ├── text_extractor.py        # PDF/text extraction
│   ├── section_parser.py        # Resume section parsing
│   ├── skill_extractor.py       # NLP-based skill extraction
│   ├── skill_normalizer.py      # Skill normalization
│   ├── scoring_engine.py        # ATS & semantic scoring
│   ├── feature_generator.py     # ML feature engineering
│   ├── classifier.py            # Job category classification
│   ├── clustering_engine.py     # K-Means clustering
│   ├── association_miner.py     # Apriori association mining
│   ├── evaluation_module.py     # Performance & fairness evaluation
│   ├── resume_processor.py      # Pipeline orchestrator
│   └── models.py                # Data models & schemas
├── tests/                        # Unit & integration tests
├── config/                       # Configuration files
│   ├── config.yaml              # System configuration
│   ├── skill_aliases.json       # Skill normalization mappings
│   └── job_categories.json      # Valid job categories
├── output/                       # Generated outputs
│   ├── csv_structured/          # Structured resume JSONs (CSV)
│   ├── pdf_structured/          # Structured resume JSONs (PDF)
│   ├── models/                  # Trained ML models
│   └── reports/                 # Evaluation reports
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_skill_extractor.py
```

The submission was checked with the main CLI integration tests and targeted
evaluation/association-mining tests listed in `INTEGRATION_TESTS_GUIDE.md`.

## Latest Real-Data Results

Artifacts from the latest run are stored in:
- `output/models/`
- `output/reports/evaluation_report.json`
- `output/reports/association_rules.json`
- `output/reports/validation_report.json`

### Classification Evaluation

Config used:
- test split: `0.2`
- random state: `42`

Observed metrics on the real dataset:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Baseline: TF-IDF + Logistic Regression | 0.6258 | 0.5468 |
| Proposed: Skill Features + Random Forest | 0.2857 | 0.2301 |

Important note:
- In the current implementation, the baseline model outperformed the proposed skill-feature model on the Kaggle dataset

Proposed-model fairness analysis:
- Mean per-category F1: `0.2301`
- F1 standard deviation: `0.1837`
- Flagged low-performing categories: `ADVOCATE`, `AGRICULTURE`, `ARTS`, `AUTOMOBILE`, `BPO`, `CONSULTANT`

Top proposed-model categories by F1:
- `CHEF`: `0.6531`
- `ACCOUNTANT`: `0.5161`
- `TEACHER`: `0.4364`
- `BUSINESS-DEVELOPMENT`: `0.4314`
- `INFORMATION-TECHNOLOGY`: `0.3939`

### Association Mining

The latest rule output contains 2 rules:
- `{state} => {name city}` with support `0.1450`, confidence `0.5854`, lift `2.2952`
- `{name city} => {state}` with support `0.1450`, confidence `0.5687`, lift `2.2952`

Important note:
- The current mining output is dominated by resume header/location artifacts rather than clean skill bundles, which indicates additional text cleanup is still needed before association rules are presentation-ready

### Cross-Source Validation

Validation compares successfully processed CSV/PDF pairs by `resume_id`.

Observed metrics:
- Samples compared: `2483`
- Text similarity: `0.9979 +/- 0.0172`
- Skill overlap: `0.7575 +/- 0.1633`
- Extraction accuracy: `0.8777`

Interpretation:
- PDF text extraction is very close to the CSV text source
- Skill overlap is materially lower than text similarity, so the skill extraction pipeline is more lossy than raw text extraction

## Current Scope and Limitations

- The repo name uses "multi-agent", but the shipped implementation is a modular in-process Python pipeline rather than a runtime system of independent agents
- `src/clustering_engine.py` is implemented and tested, but clustering is not currently exposed through the CLI
- The scoring engine exists, but normal CLI processing still writes structured resumes with `scores: null`
- The real-data results above come from the current checked-in pipeline and should be used instead of earlier placeholder claims

## Documentation

- **README**: Project overview and setup instructions
- **Real Data Results**: `REAL_DATA_RESULTS.md`
- **CLI Reference**: `CLI_REFERENCE.md`
- **API Documentation**: `API_DOCUMENTATION.md`
- **Integration Tests Guide**: `INTEGRATION_TESTS_GUIDE.md`

## Contributors

**CSE 572 Group 1**:
- Luan Nguyen (ltnguy58@asu.edu)
- Karthik Ponugoti (kponugo2@asu.edu)
- Krish Naik (knaik13@asu.edu)
- Kiran Kamalakar (kkamala1@asu.edu)
- Sai Rithwik Reddy Chirra (schirra7@asu.edu)

## License

This is an academic project for CSE 572: Data Mining at Arizona State University.

## Acknowledgments

- Dataset: Kaggle Resume Dataset by Snehaan Bhawal
- Course: CSE 572 - Data Mining, Arizona State University
- Inspired by research on semantic matching in ATS systems and fairness in algorithmic hiring

## Contact

For questions or collaboration opportunities, please contact the project team members listed above.

---

**Note**: This project demonstrates the application of data mining techniques to real-world resume screening challenges, addressing limitations of traditional keyword-based ATS systems through semantic understanding and fairness analysis.
