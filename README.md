# Multi-Agent Resume Screening & Skill Mining System

**CSE 572: Data Mining - Academic Project**

An intelligent NLP-powered resume processing pipeline that extracts, normalizes, and analyzes resume data to improve candidate evaluation beyond traditional keyword-based Applicant Tracking Systems (ATS).

## 🎯 Project Overview

This system processes 2,400+ resumes across 25 job categories using a multi-agent pipeline that:
- Extracts structured information from PDF and CSV resume sources
- Performs semantic analysis using transformer-based embeddings
- Enables data mining tasks including classification, clustering, and association rule mining
- Provides comprehensive fairness analysis across job categories

## ✨ Key Features

### Multi-Agent Pipeline
- **Text Extraction Agent**: Converts PDF resumes to clean text using pdfplumber
- **Section Parser Agent**: Divides resumes into logical sections (Skills, Experience, Education, Projects)
- **Skill Extractor Agent**: Uses spaCy NLP to identify explicit and implicit skills
- **Skill Normalizer Agent**: Standardizes skill variations using fuzzy matching (RapidFuzz)
- **Scoring Agents**: Dual scoring system (ATS keyword matching + semantic similarity)

### Machine Learning Capabilities
- **Classification**: Predicts job categories using Random Forest on skill features
- **Clustering**: Groups similar candidates using K-Means clustering
- **Association Mining**: Discovers frequently co-occurring skills using Apriori algorithm
- **Fairness Analysis**: Evaluates performance equity across job categories

### Dual Data Source Support
- **CSV Processing**: Fast processing using pre-extracted text from `archive/Resume/Resume.csv`
- **PDF Processing**: Full extraction pipeline validation from `archive/data/data/` organized by job categories
- **Cross-Validation**: Compares PDF extraction accuracy against CSV ground truth

## 🛠️ Technology Stack

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

## 📦 Installation

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
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Usage

### Command-Line Interface

The system provides a comprehensive CLI with multiple commands:

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
- ML hyperparameters (n_clusters, min_support, min_confidence)

## 📊 Project Structure

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

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_skill_extractor.py
```

**Test Coverage**: 502/504 tests passing (99.6% pass rate)

## 📈 Results & Performance

### Classification Performance
- **Baseline Model** (TF-IDF + Logistic Regression): Accuracy & Macro-F1 scores
- **Proposed Model** (Skill Features + Random Forest): Improved performance over baseline

### Clustering Analysis
- K-Means clustering with configurable k (default: 10)
- Silhouette score evaluation for cluster quality

### Association Rules
- Discovered frequent skill co-occurrence patterns
- Example rules:
  - `{Machine Learning, Python} => {SQL}` (data science cluster)
  - `{React, JavaScript} => {Node.js, HTML, CSS}` (web development cluster)

### Fairness Analysis
- Per-category F1 score distribution
- Identification of under-performing categories
- Variance analysis across job categories

## 📝 Documentation

- **README**: Project overview and setup instructions
- **CLI Reference**: `CLI_REFERENCE.md`
- **API Documentation**: `API_DOCUMENTATION.md`
- **Integration Tests Guide**: `INTEGRATION_TESTS_GUIDE.md`

## 🤝 Contributors

**CSE 572 Group 1**:
- Luan Nguyen (ltnguy58@asu.edu)
- Karthik Ponugoti (kponugo2@asu.edu)
- Krish Naik (knaik13@asu.edu)
- Kiran Kamalakar (kkamala1@asu.edu)
- Sai Rithwik Reddy Chirra (schirra7@asu.edu)

## 📄 License

This is an academic project for CSE 572: Data Mining at Arizona State University.

## 🙏 Acknowledgments

- Dataset: Kaggle Resume Dataset by Snehaan Bhawal
- Course: CSE 572 - Data Mining, Arizona State University
- Inspired by research on semantic matching in ATS systems and fairness in algorithmic hiring

## 📧 Contact

For questions or collaboration opportunities, please contact the project team members listed above.

---

**Note**: This project demonstrates the application of data mining techniques to real-world resume screening challenges, addressing limitations of traditional keyword-based ATS systems through semantic understanding and fairness analysis.
