# Smart Resume Screening System - CLI Reference

## Overview

The Smart Resume Screening System provides a command-line interface for processing resumes from CSV and PDF sources, training machine learning models, and generating analysis reports.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Show all available commands
python main.py --help

# Show help for a specific command
python main.py train --help
```

## Commands

### 1. process-csv

Extract and structure resumes from CSV file.

**Usage:**
```bash
python main.py process-csv [OPTIONS]
```

**Options:**
- `--csv-file PATH`: Path to CSV file (default: `archive/Resume/Resume.csv`)
- `--config PATH`: Configuration file (default: `config/config.yaml`)
- `--output-dir PATH`: Output directory (default: `output`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py process-csv --csv-file archive/Resume/Resume.csv
```

**Output:**
- Structured JSON files in `output/csv_structured/`
- Each resume saved as `{resume_id}.json`

**What it does:**
1. Loads resume data from CSV (ID, Resume_str, Resume_html, Category)
2. Processes Resume_str text (skips PDF extraction)
3. Extracts skills using NLP
4. Normalizes skills using alias dictionary
5. Saves structured JSON for each resume

---

### 2. process-pdf

Extract and structure resumes from PDF archive directory.

**Usage:**
```bash
python main.py process-pdf [OPTIONS]
```

**Options:**
- `--pdf-dir PATH`: Path to PDF archive directory (default: `archive/data/data`)
- `--config PATH`: Configuration file (default: `config/config.yaml`)
- `--output-dir PATH`: Output directory (default: `output`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py process-pdf --pdf-dir archive/data/data
```

**Output:**
- Structured JSON files in `output/pdf_structured/`
- Each resume saved as `{resume_id}.json`

**What it does:**
1. Scans PDF archive organized by job category folders
2. Extracts text from each PDF using pdfplumber
3. Parses sections (Skills, Experience, Education, Projects)
4. Extracts and normalizes skills
5. Saves structured JSON for each resume

---

### 3. train

Train ML models on CSV data.

**Usage:**
```bash
python main.py train [OPTIONS]
```

**Options:**
- `--csv-file PATH`: Path to CSV file for training (default: `archive/Resume/Resume.csv`)
- `--config PATH`: Configuration file (default: `config/config.yaml`)
- `--output-dir PATH`: Output directory (default: `output`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py train --csv-file archive/Resume/Resume.csv
```

**Output:**
- `output/models/classifier.pkl`: Trained classifier (baseline + proposed)
- `output/models/feature_generator.pkl`: Feature generator with vocabulary
- `output/models/vocabulary.json`: Skill vocabulary

**What it does:**
1. Loads and processes CSV resume data
2. Generates feature matrix from normalized skills
3. Trains baseline model (TF-IDF + Logistic Regression)
4. Trains proposed model (Skill Features + Random Forest)
5. Saves trained models and vocabulary

**Models:**
- **Baseline**: TF-IDF vectorization + Logistic Regression
- **Proposed**: Binary skill features + Random Forest (100 trees, max_depth=20)

---

### 4. evaluate

Evaluate trained models and generate performance reports.

**Usage:**
```bash
python main.py evaluate [OPTIONS]
```

**Options:**
- `--csv-file PATH`: Path to CSV file for evaluation (default: `archive/Resume/Resume.csv`)
- `--config PATH`: Configuration file (default: `config/config.yaml`)
- `--output-dir PATH`: Output directory (default: `output`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py evaluate --csv-file archive/Resume/Resume.csv
```

**Output:**
- `output/reports/evaluation_report.json`: Comprehensive evaluation report
- Console summary with metrics

**What it does:**
1. Loads trained models from `output/models/`
2. Processes test data and generates predictions
3. Calculates metrics: accuracy, macro F1, per-class F1
4. Compares baseline vs proposed model
5. Analyzes fairness across job categories
6. Generates detailed report

**Metrics:**
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1 score across all categories
- **Per-class F1**: F1 score for each job category
- **Fairness**: Variance and flagged categories

---

### 5. mine

Run association mining to discover frequently co-occurring skills.

**Usage:**
```bash
python main.py mine [OPTIONS]
```

**Options:**
- `--csv-file PATH`: Path to CSV file (default: `archive/Resume/Resume.csv`)
- `--config PATH`: Configuration file (default: `config/config.yaml`)
- `--output-dir PATH`: Output directory (default: `output`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py mine --csv-file archive/Resume/Resume.csv
```

**Output:**
- `output/reports/association_rules.json`: Discovered association rules
- Console display of top rules by lift

**What it does:**
1. Loads resume data and extracts normalized skills
2. Transforms resumes to transactions (skill sets)
3. Applies Apriori algorithm to find frequent itemsets
4. Generates association rules with support, confidence, lift
5. Saves rules sorted by lift

**Rule Format:**
```json
{
  "antecedents": ["Python", "Machine Learning"],
  "consequents": ["Data Science"],
  "support": 0.25,
  "confidence": 0.80,
  "lift": 2.5
}
```

**Interpretation:**
- **Support**: How often the skill combination appears
- **Confidence**: Probability of consequent given antecedent
- **Lift**: How much more likely consequent is with antecedent (>1 = positive correlation)

---

### 6. validate

Run cross-source validation between CSV and PDF data.

**Usage:**
```bash
python main.py validate [OPTIONS]
```

**Options:**
- `--csv-file PATH`: Path to CSV file (default: `archive/Resume/Resume.csv`)
- `--pdf-dir PATH`: Path to PDF archive directory (default: `archive/data/data`)
- `--config PATH`: Configuration file (default: `config/config.yaml`)
- `--output-dir PATH`: Output directory (default: `output`)
- `--log-level LEVEL`: Logging level (default: `INFO`)

**Example:**
```bash
python main.py validate --csv-file archive/Resume/Resume.csv --pdf-dir archive/data/data
```

**Output:**
- `output/reports/validation_report.json`: Validation metrics
- Console summary with similarity scores

**What it does:**
1. Loads resume data from both CSV and PDF sources
2. Compares text extraction accuracy
3. Calculates skill overlap between sources
4. Generates validation report

**Metrics:**
- **Text Similarity**: Mean and std dev of text similarity scores
- **Skill Overlap**: Mean and std dev of skill overlap percentages
- **Extraction Accuracy**: Overall extraction pipeline accuracy

---

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# PDF extraction
pdf_extractor: pdfplumber  # or pypdf

# NLP models
nlp_model: en_core_web_sm
embedding_model: all-MiniLM-L6-v2

# Skill normalization
fuzzy_threshold: 85
alias_dict_path: config/skill_aliases.json

# ML parameters
n_clusters: 10
min_support: 0.1
min_confidence: 0.5
test_size: 0.2
random_state: 42
```

---

## Workflow Examples

### Basic Workflow

```bash
# 1. Process CSV data
python main.py process-csv

# 2. Train models
python main.py train

# 3. Evaluate models
python main.py evaluate
```

### Complete Analysis

```bash
# Process both data sources
python main.py process-csv
python main.py process-pdf

# Train and evaluate
python main.py train
python main.py evaluate

# Mine associations
python main.py mine

# Validate extraction
python main.py validate
```

### Custom Configuration

```bash
# Use custom config and output directory
python main.py --config my_config.yaml --output-dir results train

# Enable debug logging
python main.py --log-level DEBUG evaluate
```

---

## Troubleshooting

### Common Issues

**1. spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**2. CSV file not found**
```bash
# Specify correct path
python main.py process-csv --csv-file path/to/Resume.csv
```

**3. Models not found (during evaluate)**
```bash
# Train models first
python main.py train
```

**4. Memory issues with large datasets**
```bash
# Process in smaller batches or increase system memory
```

### Logging

Enable debug logging for troubleshooting:
```bash
python main.py --log-level DEBUG <command>
```

---

## Output Files

### Structured Resumes (JSON)

```json
{
  "resume_id": "16852973",
  "job_category": "HR",
  "sections": {
    "skills": "...",
    "experience": "...",
    "education": "...",
    "projects": "...",
    "raw_text": "..."
  },
  "skills": {
    "explicit_skills": ["Python", "Machine Learning"],
    "implicit_skills": ["Data Analysis", "SQL"]
  },
  "normalized_skills": ["Python", "Machine Learning", "Data Analysis", "SQL"],
  "scores": null,
  "metadata": {
    "file_path": "archive/Resume/Resume.csv",
    "processed_date": "2025-01-01T12:00:00",
    "processing_time_ms": 1234
  }
}
```

### Evaluation Report (JSON)

```json
{
  "classification": {
    "accuracy": 0.2857,
    "macro_f1": 0.2301,
    "per_class_f1": {...}
  },
  "model_comparison": {
    "baseline": {
      "accuracy": 0.6258,
      "macro_f1": 0.5468,
      "per_class_f1": {...}
    },
    "proposed": {
      "accuracy": 0.2857,
      "macro_f1": 0.2301,
      "per_class_f1": {...}
    },
    "improvements": {
      "accuracy": -0.3400,
      "macro_f1": -0.3167
    }
  },
  "fairness": {
    "mean_f1": 0.2301,
    "f1_variance": 0.0338,
    "f1_std": 0.1837,
    "flagged_categories": ["ADVOCATE", "AGRICULTURE", "ARTS", "AUTOMOBILE", "BPO", "CONSULTANT"]
  }
}
```

---

## Performance Tips

1. **Use CSV data for training**: Faster than PDF processing
2. **Process PDFs in batches**: For large archives
3. **Adjust thresholds**: Lower min_support/min_confidence for more rules
4. **Enable caching**: Reuse processed data when possible
5. **Monitor memory**: Large datasets may require more RAM

---

## Support

For issues or questions:
1. Check this reference guide
2. Review README.md
3. Check logs with `--log-level DEBUG`
4. Review design.md for architecture details
