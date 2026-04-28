"""
Integration tests for Classifier component with CSV data.

This test demonstrates the complete workflow:
1. Load CSV data
2. Process resumes to extract skills
3. Generate feature vectors
4. Train both baseline and proposed models
5. Make predictions
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.classifier import Classifier
from src.resume_processor import ResumeProcessor
from src.feature_generator import FeatureGenerator
from src.models import ProcessorConfig


@pytest.fixture
def sample_csv_data(tmp_path):
    """Create a small sample CSV file for testing."""
    csv_path = tmp_path / "sample_resumes.csv"
    
    # Create sample data
    data = {
        'ID': ['1001', '1002', '1003', '1004', '1005'],
        'Resume_str': [
            """
            SKILLS
            Python, Java, SQL, Machine Learning, Data Science
            
            EXPERIENCE
            Software Engineer at Tech Corp
            Developed ML models and data pipelines
            """,
            """
            SKILLS
            Accounting, Excel, Financial Reporting, Tax
            
            EXPERIENCE
            Senior Accountant at Finance Inc
            Managed financial statements and tax compliance
            """,
            """
            SKILLS
            Legal Research, Contract Law, Litigation
            
            EXPERIENCE
            Associate Attorney at Law Firm
            Handled contract disputes and litigation cases
            """,
            """
            SKILLS
            JavaScript, React, Node.js, Web Development
            
            EXPERIENCE
            Frontend Developer at Web Solutions
            Built responsive web applications
            """,
            """
            SKILLS
            Budgeting, Financial Analysis, QuickBooks
            
            EXPERIENCE
            Financial Analyst at Corp Finance
            Analyzed budgets and financial performance
            """
        ],
        'Resume_html': [''] * 5,
        'Category': ['IT', 'ACCOUNTANT', 'ADVOCATE', 'IT', 'ACCOUNTANT']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


def test_classifier_with_csv_workflow(sample_csv_data):
    """Test complete workflow: CSV -> Features -> Training -> Prediction."""
    
    # Initialize components
    config = ProcessorConfig()
    processor = ResumeProcessor(config)
    feature_gen = FeatureGenerator()
    classifier = Classifier()
    
    # Step 1: Process CSV data
    structured_resumes = processor.process_csv_data(sample_csv_data)
    
    assert len(structured_resumes) == 5
    assert all(len(r.normalized_skills) > 0 for r in structured_resumes)
    
    # Step 2: Generate feature matrix
    X, vocabulary = feature_gen.generate_feature_matrix(structured_resumes)
    
    assert X.shape[0] == 5  # 5 resumes
    assert X.shape[1] == len(vocabulary)  # Number of unique skills
    assert len(vocabulary) > 0
    
    # Step 3: Prepare labels
    y = np.array([r.job_category for r in structured_resumes])
    
    assert len(y) == 5
    assert set(y) == {'IT', 'ACCOUNTANT', 'ADVOCATE'}
    
    # Step 4: Train proposed model (skill features + Random Forest)
    classifier.train_proposed(X, y)
    
    assert classifier.proposed_model is not None
    assert classifier.classes_ is not None
    
    # Step 5: Train baseline model (TF-IDF + Logistic Regression)
    resume_texts = [r.sections.raw_text for r in structured_resumes]
    classifier.train_baseline(resume_texts, y)
    
    assert classifier.baseline_model is not None
    assert classifier.baseline_vectorizer is not None
    
    # Step 6: Make predictions with proposed model
    predictions_proposed = classifier.predict(X, model_type="proposed")
    
    assert len(predictions_proposed) == 5
    assert all(pred in ['IT', 'ACCOUNTANT', 'ADVOCATE'] for pred in predictions_proposed)
    
    # Step 7: Make predictions with baseline model
    predictions_baseline = classifier.predict(
        None,
        model_type="baseline",
        resume_texts=resume_texts
    )
    
    assert len(predictions_baseline) == 5
    assert all(pred in ['IT', 'ACCOUNTANT', 'ADVOCATE'] for pred in predictions_baseline)
    
    # Step 8: Get prediction probabilities
    proba_proposed = classifier.predict_proba(X, model_type="proposed")
    
    assert proba_proposed.shape == (5, 3)  # 5 samples, 3 classes
    assert np.allclose(proba_proposed.sum(axis=1), 1.0)
    
    proba_baseline = classifier.predict_proba(
        None,
        model_type="baseline",
        resume_texts=resume_texts
    )
    
    assert proba_baseline.shape == (5, 3)
    assert np.allclose(proba_baseline.sum(axis=1), 1.0)
    
    print("\n[PASS] Complete workflow test passed!")
    print(f"  - Processed {len(structured_resumes)} resumes")
    print(f"  - Vocabulary size: {len(vocabulary)} skills")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Classes: {classifier.classes_}")
    print(f"  - Proposed predictions: {predictions_proposed}")
    print(f"  - Baseline predictions: {predictions_baseline}")


def test_feature_generator_with_csv(sample_csv_data):
    """Test FeatureGenerator.load_csv_features() method."""
    
    config = ProcessorConfig()
    processor = ResumeProcessor(config)
    feature_gen = FeatureGenerator()
    
    # Load features directly from CSV
    X, vocabulary, structured_resumes = feature_gen.load_csv_features(
        sample_csv_data,
        processor
    )
    
    assert X.shape[0] == 5
    assert X.shape[1] == len(vocabulary)
    assert len(structured_resumes) == 5
    
    # Verify binary features
    assert X.dtype == np.int8
    assert np.all((X == 0) | (X == 1))
    
    print("\n[PASS] CSV feature loading test passed!")
    print(f"  - Feature matrix: {X.shape}")
    print(f"  - Vocabulary: {len(vocabulary)} skills")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
