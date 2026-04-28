# Smart Resume Screening System - API Documentation

## Overview

This document provides comprehensive API documentation for the Smart Resume Screening System, covering all public classes, methods, and data models. The system supports dual data sources (CSV and PDF) for flexible resume processing and machine learning workflows.

## Table of Contents

- [Data Models](#data-models)
- [Core Components](#core-components)
- [Processing Pipeline](#processing-pipeline)
- [Machine Learning Components](#machine-learning-components)
- [Evaluation Components](#evaluation-components)
- [Configuration](#configuration)
- [Examples](#examples)
- [Error Handling](#error-handling)

## Data Models

### ResumeSections

Represents the parsed sections of a resume.

```python
@dataclass
class ResumeSections:
    skills: str          # Text content from the Skills section
    experience: str      # Text content from the Experience section
    education: str       # Text content from the Education section
    projects: str        # Text content from the Projects section
    raw_text: str        # Complete unprocessed resume text
```

**Usage Example**:
```python
sections = ResumeSections(
    skills="Python, JavaScript, SQL",
    experience="5 years as Software Engineer...",
    education="BS Computer Science...",
    projects="Built web application...",
    raw_text="Full resume text..."
)
```

### SkillSet

Represents extracted skills from a resume with explicit/implicit distinction.

```python
@dataclass
class SkillSet:
    explicit_skills: List[str]  # Skills directly listed in Skills section
    implicit_skills: List[str]  # Skills inferred from Experience/Projects
    
    def all_skills(self) -> List[str]:
        """Combine explicit and implicit skills into a single list."""
        return self.explicit_skills + self.implicit_skills
```

**Usage Example**:
```python
skill_set = SkillSet(
    explicit_skills=["Python", "SQL", "JavaScript"],
    implicit_skills=["Data Analysis", "Web Development"]
)
all_skills = skill_set.all_skills()  # Returns combined list
```

### Scores

Represents scoring metrics for a resume.

```python
@dataclass
class Scores:
    ats_score: float      # Keyword match score (0-100)
    semantic_score: float # Semantic similarity score (0-1)
```

**Usage Example**:
```python
scores = Scores(ats_score=75.5, semantic_score=0.82)
```

### StructuredResume

Complete structured representation of a processed resume.

```python
@dataclass
class StructuredResume:
    resume_id: str
    job_category: str
    sections: ResumeSections
    skills: SkillSet
    normalized_skills: List[str]
    scores: Optional[Scores]
    metadata: ResumeMetadata
    
    def to_json(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        
    @classmethod
    def from_json(cls, data: dict) -> 'StructuredResume':
        """Create from JSON dictionary."""
```

**Usage Example**:
```python
# Create structured resume
resume = StructuredResume(
    resume_id="12345",
    job_category="SOFTWARE_ENGINEER",
    sections=sections,
    skills=skill_set,
    normalized_skills=["Python", "SQL", "JavaScript"],
    scores=scores,
    metadata=metadata
)

# Convert to JSON
json_data = resume.to_json()

# Create from JSON
resume_from_json = StructuredResume.from_json(json_data)
```

### Configuration Models

#### ProcessorConfig

Configuration for the resume processor.

```python
@dataclass
class ProcessorConfig:
    pdf_extractor: str = "pdfplumber"           # PDF extraction library
    nlp_model: str = "en_core_web_sm"           # spaCy model name
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    fuzzy_threshold: int = 85                   # Fuzzy matching threshold (0-100)
    alias_dict_path: str = "config/skill_aliases.json"  # Skill alias dictionary
```

#### MLConfig

Configuration for machine learning components.

```python
@dataclass
class MLConfig:
    n_clusters: int = 10          # Number of clusters for K-Means
    min_support: float = 0.1      # Minimum support for Apriori
    min_confidence: float = 0.5   # Minimum confidence for association rules
    test_size: float = 0.2        # Test set proportion (0-1)
    random_state: int = 42        # Random seed for reproducibility
```

## Core Components

### TextExtractor

Extracts and cleans text from PDF and text files.

```python
class TextExtractor:
    SUPPORTED_FORMATS = {'.pdf', '.txt'}
    
    def __init__(self):
        """Initialize the TextExtractor."""
    
    def validate_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber with pypdf fallback."""
    
    def extract_from_text(self, text_path: str) -> str:
        """Load text from plain text file."""
    
    def clean_text(self, raw_text: str) -> str:
        """Remove special characters and normalize whitespace."""
```

**Usage Example**:
```python
extractor = TextExtractor()

# Check format support
if extractor.validate_format("resume.pdf"):
    # Extract from PDF
    text = extractor.extract_from_pdf("resume.pdf")
    
    # Clean the text
    clean_text = extractor.clean_text(text)
```

**Error Handling**:
- `FileNotFoundError`: File does not exist
- `UnsupportedFormatError`: File format not supported
- `PDFExtractionError`: PDF extraction failed

### SectionParser

Divides resume text into logical sections using pattern matching.

```python
class SectionParser:
    def __init__(self, section_patterns: Optional[Dict[str, List[str]]] = None):
        """Initialize with regex patterns for section headers."""
    
    def parse_sections(self, text: str) -> ResumeSections:
        """Identify and extract all resume sections."""
    
    def extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section by name."""
```

**Usage Example**:
```python
parser = SectionParser()
sections = parser.parse_sections(resume_text)

# Access individual sections
skills_text = sections.skills
experience_text = sections.experience
```

**Section Detection**:
- **Skills**: `(?i)(skills?|technical skills?|core competencies)`
- **Experience**: `(?i)(experience|work history|employment)`
- **Education**: `(?i)(education|academic background|qualifications)`
- **Projects**: `(?i)(projects?|portfolio)`

### SkillExtractor

Extracts skills using NLP and Named Entity Recognition.

```python
class SkillExtractor:
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """Initialize with spaCy NLP model."""
    
    def extract_explicit_skills(self, skills_section: str) -> List[str]:
        """Extract skills from Skills section using NER and noun phrases."""
    
    def extract_implicit_skills(self, experience: str, projects: str) -> List[str]:
        """Infer skills from Experience and Projects sections."""
    
    def extract_all_skills(self, sections: ResumeSections) -> SkillSet:
        """Extract both explicit and implicit skills."""
```

**Usage Example**:
```python
extractor = SkillExtractor(nlp_model="en_core_web_sm")
skill_set = extractor.extract_all_skills(sections)

print(f"Explicit skills: {skill_set.explicit_skills}")
print(f"Implicit skills: {skill_set.implicit_skills}")
print(f"All skills: {skill_set.all_skills()}")
```

**Error Handling**:
- `ModelLoadError`: spaCy model failed to load
- `SkillExtractionError`: Skill extraction process failed

### SkillNormalizer

Standardizes skill variations using fuzzy matching and alias dictionaries.

```python
class SkillNormalizer:
    def __init__(self, alias_dict: Dict[str, str], fuzzy_threshold: int = 85):
        """Initialize with alias dictionary and fuzzy matching threshold."""
    
    def load_alias_dictionary(self, dict_path: str) -> Dict[str, str]:
        """Load skill alias mappings from JSON file."""
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize a single skill to canonical form."""
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize a list of skills."""
```

**Usage Example**:
```python
# Load alias dictionary
normalizer = SkillNormalizer(alias_dict={}, fuzzy_threshold=85)
alias_dict = normalizer.load_alias_dictionary("config/skill_aliases.json")

# Create normalizer with loaded dictionary
normalizer = SkillNormalizer(alias_dict=alias_dict, fuzzy_threshold=85)

# Normalize skills
raw_skills = ["js", "react.js", "ml", "python3"]
normalized = normalizer.normalize_skills(raw_skills)
# Result: ["JavaScript", "React", "Machine Learning", "Python"]
```

**Normalization Process**:
1. **Exact Match**: Check alias dictionary for exact match
2. **Fuzzy Match**: Use RapidFuzz to find similar canonical skills (threshold: 85%)
3. **Fallback**: Return original skill in lowercase if no match found

### ScoringEngine

Calculates ATS and semantic similarity scores.

```python
class ScoringEngine:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model."""
    
    def calculate_ats_score(self, resume_skills: List[str], job_requirements: List[str]) -> float:
        """Calculate keyword match percentage (0-100)."""
    
    def calculate_semantic_score(self, resume_skills: List[str], job_requirements: List[str]) -> float:
        """Calculate cosine similarity using embeddings (0-1)."""
    
    def calculate_both_scores(self, resume_skills: List[str], job_requirements: List[str]) -> Scores:
        """Calculate both ATS and semantic scores."""
```

**Usage Example**:
```python
scoring_engine = ScoringEngine()

resume_skills = ["Python", "Machine Learning", "SQL"]
job_requirements = ["Python", "Data Science", "SQL", "TensorFlow"]

# Calculate individual scores
ats_score = scoring_engine.calculate_ats_score(resume_skills, job_requirements)
semantic_score = scoring_engine.calculate_semantic_score(resume_skills, job_requirements)

# Calculate both scores
scores = scoring_engine.calculate_both_scores(resume_skills, job_requirements)
```

**Scoring Methods**:
- **ATS Score**: `(matched_skills / total_required_skills) * 100`
- **Semantic Score**: Cosine similarity between skill embeddings

## Processing Pipeline

### ResumeProcessor

Orchestrates the complete resume processing pipeline with dual data source support.

```python
class ResumeProcessor:
    def __init__(self, config: ProcessorConfig):
        """Initialize processor with all component dependencies."""
    
    def process_resume(self, file_path: str, job_category: Optional[str] = None, 
                      resume_id: Optional[str] = None) -> StructuredResume:
        """Process a single resume file (PDF or text)."""
    
    def process_batch(self, directory: str) -> List[StructuredResume]:
        """Process all resumes in a directory."""
    
    def load_from_archive(self, archive_path: str) -> Dict[str, List[StructuredResume]]:
        """Load resumes organized by job category folders."""
    
    def load_from_csv(self, csv_path: str) -> List[Dict[str, str]]:
        """Load resume data from CSV file."""
    
    def process_csv_data(self, csv_path: str) -> List[StructuredResume]:
        """Process resume entries from CSV file."""
    
    def validate_csv_extraction(self, pdf_path: str, csv_resume_str: str, 
                               resume_id: str) -> Dict[str, any]:
        """Compare PDF extraction results against CSV Resume_str data."""
    
    def cross_validate_data_sources(self, pdf_archive_path: str, csv_path: str, 
                                   max_samples: Optional[int] = None) -> Dict[str, any]:
        """Identify discrepancies between PDF and CSV text."""
```

### CSV Data Source Processing

**CSV Structure Expected**:
```csv
ID,Resume_str,Resume_html,Category
12345,"John Doe Software Engineer...","<html>...","SOFTWARE_ENGINEER"
```

**Usage Example**:
```python
config = ProcessorConfig()
processor = ResumeProcessor(config)

# Process CSV data (fast, pre-extracted text)
csv_resumes = processor.process_csv_data("archive/Resume/Resume.csv")

# Load raw CSV data
csv_data = processor.load_from_csv("archive/Resume/Resume.csv")
```

### PDF Archive Processing

**Archive Structure Expected**:
```
archive/data/data/
├── ACCOUNTANT/
│   ├── 10554236.pdf
│   └── 10674770.pdf
├── ADVOCATE/
│   └── 10186968.pdf
└── ...
```

**Usage Example**:
```python
# Process PDF archive (slower, real extraction)
pdf_resumes = processor.load_from_archive("archive/data/data")

# Result: Dict[category, List[StructuredResume]]
for category, resumes in pdf_resumes.items():
    print(f"{category}: {len(resumes)} resumes")
```

### Cross-Source Validation

**Usage Example**:
```python
# Validate PDF extraction against CSV ground truth
validation_report = processor.cross_validate_data_sources(
    pdf_archive_path="archive/data/data",
    csv_path="archive/Resume/Resume.csv",
    max_samples=100
)

print(f"Success rate: {validation_report['success_rate']:.1%}")
print(f"Average similarity: {validation_report['average_similarity']:.3f}")
```

## Machine Learning Components

### FeatureGenerator

Converts skills to numerical feature vectors for ML.

```python
class FeatureGenerator:
    def __init__(self):
        """Initialize feature generator."""
    
    def build_vocabulary(self, all_resumes: List[StructuredResume]) -> List[str]:
        """Create vocabulary of all unique skills across dataset."""
    
    def generate_feature_vector(self, skills: List[str], vocabulary: List[str]) -> np.ndarray:
        """Convert skills to binary feature vector."""
    
    def generate_feature_matrix(self, all_resumes: List[StructuredResume]) -> Tuple[np.ndarray, List[str]]:
        """Generate feature matrix for all resumes."""
```

**Usage Example**:
```python
feature_gen = FeatureGenerator()

# Generate feature matrix from resumes
X, vocabulary = feature_gen.generate_feature_matrix(structured_resumes)

print(f"Feature matrix shape: {X.shape}")
print(f"Vocabulary size: {len(vocabulary)}")
```

### Classifier

Predicts job categories using baseline and proposed models.

```python
class Classifier:
    def __init__(self):
        """Initialize classifier with baseline and proposed models."""
    
    def train_baseline(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train TF-IDF + Logistic Regression baseline."""
    
    def train_proposed(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train skill features + Random Forest proposed model."""
    
    def predict(self, X: np.ndarray, model_type: str = "proposed") -> np.ndarray:
        """Predict job categories."""
    
    def predict_proba(self, X: np.ndarray, model_type: str = "proposed") -> np.ndarray:
        """Predict job category probabilities."""
```

**Model Specifications**:
- **Baseline**: TF-IDF + Logistic Regression (C=1.0, max_iter=1000)
- **Proposed**: Skill Features + Random Forest (n_estimators=100, max_depth=20)

**Usage Example**:
```python
classifier = Classifier()

# Train models
classifier.train_baseline(raw_texts, y_train)
classifier.train_proposed(X_train, y_train)

# Make predictions
baseline_pred = classifier.predict(X_test, model_type="baseline")
proposed_pred = classifier.predict(X_test, model_type="proposed")

# Get probabilities
probabilities = classifier.predict_proba(X_test, model_type="proposed")
```

### ClusteringEngine

Groups similar candidates using K-Means clustering.

```python
class ClusteringEngine:
    def __init__(self, n_clusters: int = 10):
        """Initialize with number of clusters."""
    
    def fit_clusters(self, X: np.ndarray) -> np.ndarray:
        """Apply K-Means clustering and return cluster labels."""
    
    def get_cluster_centroids(self) -> np.ndarray:
        """Return cluster centroids."""
    
    def get_cluster_profiles(self, vocabulary: List[str]) -> Dict[int, List[str]]:
        """Get top skills for each cluster."""
```

**Usage Example**:
```python
clustering = ClusteringEngine(n_clusters=10)

# Fit clusters
cluster_labels = clustering.fit_clusters(X)

# Get cluster profiles
profiles = clustering.get_cluster_profiles(vocabulary)

for cluster_id, top_skills in profiles.items():
    print(f"Cluster {cluster_id}: {top_skills[:5]}")
```

### AssociationMiner

Discovers frequently co-occurring skills using Apriori algorithm.

```python
class AssociationMiner:
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        """Initialize with support and confidence thresholds."""
    
    def mine_frequent_itemsets(self, transactions: List[List[str]]) -> pd.DataFrame:
        """Find frequent skill sets using Apriori."""
    
    def generate_rules(self, frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
        """Generate association rules with support, confidence, lift."""
    
    def mine_associations(self, all_resumes: List[StructuredResume]) -> List[AssociationRule]:
        """Complete association mining pipeline."""
```

**Usage Example**:
```python
miner = AssociationMiner(min_support=0.1, min_confidence=0.5)

# Mine associations
rules = miner.mine_associations(structured_resumes)

# Display top rules
for rule in sorted(rules, key=lambda r: r.lift, reverse=True)[:5]:
    print(f"{set(rule.antecedents)} => {set(rule.consequents)}")
    print(f"  Support: {rule.support:.3f}, Confidence: {rule.confidence:.3f}, Lift: {rule.lift:.3f}")
```

## Evaluation Components

### EvaluationModule

Measures model performance and fairness across data sources.

```python
class EvaluationModule:
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        """Calculate accuracy, macro-F1 for classification."""
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> ClusteringMetrics:
        """Calculate silhouette score for clustering."""
    
    def compare_models(self, baseline_metrics: ClassificationMetrics, 
                      proposed_metrics: ClassificationMetrics) -> ComparisonReport:
        """Compare baseline vs proposed model performance."""
    
    def analyze_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        job_categories: List[str]) -> FairnessReport:
        """Analyze performance across job categories."""
    
    def evaluate_extraction_pipeline(self, csv_resumes: List[StructuredResume],
                                   pdf_resumes: List[StructuredResume]) -> ExtractionValidationReport:
        """Validate PDF extraction accuracy against CSV ground truth."""
    
    def cross_source_validation(self, csv_features: np.ndarray, pdf_features: np.ndarray,
                               csv_labels: np.ndarray, pdf_labels: np.ndarray) -> CrossSourceValidationReport:
        """Compare model performance on CSV vs PDF-derived features."""
```

**Usage Example**:
```python
evaluator = EvaluationModule()

# Evaluate classification
baseline_metrics = evaluator.evaluate_classification(y_test, baseline_pred)
proposed_metrics = evaluator.evaluate_classification(y_test, proposed_pred)

# Compare models
comparison = evaluator.compare_models(baseline_metrics, proposed_metrics)

# Analyze fairness
fairness_report = evaluator.analyze_fairness(y_test, proposed_pred, job_categories)

print(f"Accuracy improvement: {comparison.accuracy_improvement:+.4f}")
print(f"Flagged categories: {fairness_report.flagged_categories}")
```

## Complete Pipeline Example

### End-to-End CSV Processing

```python
from src.models import ProcessorConfig, MLConfig
from src.resume_processor import ResumeProcessor
from src.feature_generator import FeatureGenerator
from src.classifier import Classifier
from src.evaluation_module import EvaluationModule

# Initialize configuration
config = ProcessorConfig(
    pdf_extractor="pdfplumber",
    nlp_model="en_core_web_sm",
    embedding_model="all-MiniLM-L6-v2",
    fuzzy_threshold=85
)

ml_config = MLConfig(
    n_clusters=10,
    min_support=0.1,
    min_confidence=0.5,
    test_size=0.2,
    random_state=42
)

# Process resumes from CSV
processor = ResumeProcessor(config)
structured_resumes = processor.process_csv_data("archive/Resume/Resume.csv")

# Generate features
feature_gen = FeatureGenerator()
X, vocabulary = feature_gen.generate_feature_matrix(structured_resumes)
y = np.array([resume.job_category for resume in structured_resumes])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=ml_config.test_size, random_state=ml_config.random_state
)

# Train classifier
classifier = Classifier()
classifier.train_proposed(X_train, y_train)

# Evaluate
predictions = classifier.predict(X_test)
evaluator = EvaluationModule()
metrics = evaluator.evaluate_classification(y_test, predictions)

print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"Macro F1: {metrics.macro_f1:.4f}")
```

### Cross-Source Validation Example

```python
# Process both data sources
csv_resumes = processor.process_csv_data("archive/Resume/Resume.csv")
pdf_resumes_by_cat = processor.load_from_archive("archive/data/data")
pdf_resumes = [resume for resumes in pdf_resumes_by_cat.values() for resume in resumes]

# Align matched resumes by ID for extraction validation
csv_by_id = {resume.resume_id: resume for resume in csv_resumes}
pdf_by_id = {resume.resume_id: resume for resume in pdf_resumes}
common_ids = sorted(set(csv_by_id).intersection(pdf_by_id))

csv_texts = [csv_by_id[resume_id].sections.raw_text for resume_id in common_ids]
pdf_texts = [pdf_by_id[resume_id].sections.raw_text for resume_id in common_ids]
csv_skills = [csv_by_id[resume_id].normalized_skills for resume_id in common_ids]
pdf_skills = [pdf_by_id[resume_id].normalized_skills for resume_id in common_ids]

# Validate extraction pipeline
extraction_report = evaluator.evaluate_extraction_pipeline(
    csv_texts, pdf_texts, csv_skills, pdf_skills
)

print(f"Extraction accuracy: {extraction_report.extraction_accuracy:.4f}")
print(f"Text similarity: {extraction_report.text_similarity_mean:.4f} +/- {extraction_report.text_similarity_std:.4f}")

# Cross-validate model performance
csv_X, csv_vocab = feature_gen.generate_feature_matrix(csv_resumes)
pdf_X, pdf_vocab = feature_gen.generate_feature_matrix(pdf_resumes)

# Train on CSV, test on both
classifier.train_proposed(csv_X, csv_y)
csv_pred = classifier.predict(csv_X_test)
pdf_pred = classifier.predict(pdf_X_test)

csv_metrics = evaluator.evaluate_classification(csv_y_test, csv_pred)
pdf_metrics = evaluator.evaluate_classification(pdf_y_test, pdf_pred)

print(f"CSV accuracy: {csv_metrics.accuracy:.4f}")
print(f"PDF accuracy: {pdf_metrics.accuracy:.4f}")
```

## Error Handling

### Exception Hierarchy

```python
# Base exceptions
class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""

class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""

class ModelLoadError(Exception):
    """Raised when spaCy model fails to load."""

class SkillExtractionError(Exception):
    """Raised when skill extraction process fails."""

class InsufficientDataError(Exception):
    """Raised when dataset is too small for ML training."""

class EmbeddingError(Exception):
    """Raised when sentence embedding calculation fails."""
```

### Error Handling Patterns

```python
try:
    # Process resume
    structured_resume = processor.process_resume("resume.pdf")
except FileNotFoundError:
    print("Resume file not found")
except UnsupportedFormatError:
    print("File format not supported")
except PDFExtractionError as e:
    print(f"PDF extraction failed: {e}")
except SkillExtractionError as e:
    print(f"Skill extraction failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Logging

All components use Python's `logging` module:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component loggers
logger = logging.getLogger(__name__)
```

**Log Levels**:
- `DEBUG`: Detailed extraction results, intermediate values
- `INFO`: Successful processing, pipeline stages
- `WARNING`: Recoverable errors, missing sections, fallback usage
- `ERROR`: Processing failures, model errors

## Performance Considerations

### Memory Usage

- **Large Datasets**: Use batch processing for datasets > 1000 resumes
- **Feature Matrices**: Consider sparse matrices for large vocabularies
- **Model Caching**: Load NLP models once and reuse

### Processing Speed

- **CSV vs PDF**: CSV processing is ~10x faster than PDF extraction
- **Parallel Processing**: Enable in config for batch processing
- **Model Selection**: Baseline models train faster than proposed models

### Optimization Tips

```python
# Batch processing for large datasets
def process_large_dataset(csv_path: str, batch_size: int = 100):
    processor = ResumeProcessor(config)
    
    # Process in batches to manage memory
    all_resumes = []
    csv_data = processor.load_from_csv(csv_path)
    
    for i in range(0, len(csv_data), batch_size):
        batch = csv_data[i:i + batch_size]
        batch_resumes = processor.process_csv_batch(batch)
        all_resumes.extend(batch_resumes)
        
        # Optional: Save intermediate results
        if i % (batch_size * 10) == 0:
            save_checkpoint(all_resumes, f"checkpoint_{i}.json")
    
    return all_resumes
```

## Version Compatibility

- **Python**: 3.9+
- **spaCy**: 3.x
- **scikit-learn**: 1.0+
- **sentence-transformers**: 2.0+
- **pandas**: 1.3+
- **numpy**: 1.20+

## Migration Guide

### From Single Source to Dual Source

```python
# Old single-source approach
processor = ResumeProcessor(config)
resumes = processor.process_batch("pdf_directory")

# New dual-source approach
# For training (fast)
csv_resumes = processor.process_csv_data("archive/Resume/Resume.csv")

# For validation (comprehensive)
pdf_resumes = processor.load_from_archive("archive/data/data")

# Cross-validation
validation_report = processor.cross_validate_data_sources(
    pdf_archive_path="archive/data/data",
    csv_path="archive/Resume/Resume.csv"
)
```

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Status**: Complete

For additional support, see the main [README.md](README.md) file or check the system logs for detailed error information.
