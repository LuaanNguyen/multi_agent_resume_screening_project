"""Basic tests for the Classifier component."""

import pytest
import numpy as np
from src.classifier import Classifier


def test_classifier_initialization():
    """Test that Classifier initializes correctly."""
    classifier = Classifier()
    
    assert classifier.baseline_vectorizer is None
    assert classifier.baseline_model is None
    assert classifier.proposed_model is None
    assert classifier.classes_ is None


def test_train_baseline():
    """Test baseline model training with TF-IDF + Logistic Regression."""
    classifier = Classifier()
    
    # Sample resume texts
    resume_texts = [
        "Python Java SQL database programming",
        "Accounting finance Excel budgeting",
        "Legal contracts law litigation",
        "Python machine learning data science",
        "Tax accounting financial reporting"
    ]
    
    # Labels
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
    
    # Train baseline model
    classifier.train_baseline(resume_texts, y_train)
    
    # Verify model is trained
    assert classifier.baseline_vectorizer is not None
    assert classifier.baseline_model is not None
    assert classifier.classes_ is not None
    assert len(classifier.classes_) == 3  # IT, ACCOUNTANT, ADVOCATE


def test_train_proposed():
    """Test proposed model training with skill features + Random Forest."""
    classifier = Classifier()
    
    # Sample binary skill feature vectors (5 resumes, 10 skills)
    X_train = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # IT skills
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Accounting skills
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # Legal skills
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # IT skills
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Accounting skills
    ])
    
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
    
    # Train proposed model
    classifier.train_proposed(X_train, y_train)
    
    # Verify model is trained
    assert classifier.proposed_model is not None
    assert classifier.classes_ is not None
    assert len(classifier.classes_) == 3


def test_predict_baseline():
    """Test baseline model predictions."""
    classifier = Classifier()
    
    # Training data
    resume_texts = [
        "Python Java SQL database programming",
        "Accounting finance Excel budgeting",
        "Legal contracts law litigation"
    ]
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
    
    classifier.train_baseline(resume_texts, y_train)
    
    # Test predictions
    test_texts = ["Python programming", "Finance accounting"]
    predictions = classifier.predict(None, model_type="baseline", resume_texts=test_texts)
    
    assert len(predictions) == 2
    assert predictions[0] in ["IT", "ACCOUNTANT", "ADVOCATE"]
    assert predictions[1] in ["IT", "ACCOUNTANT", "ADVOCATE"]


def test_predict_proposed():
    """Test proposed model predictions."""
    classifier = Classifier()
    
    # Training data
    X_train = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0]
    ])
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
    
    classifier.train_proposed(X_train, y_train)
    
    # Test predictions
    X_test = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0]
    ])
    predictions = classifier.predict(X_test, model_type="proposed")
    
    assert len(predictions) == 2
    assert predictions[0] in ["IT", "ACCOUNTANT", "ADVOCATE"]
    assert predictions[1] in ["IT", "ACCOUNTANT", "ADVOCATE"]


def test_predict_proba_baseline():
    """Test baseline model probability predictions."""
    classifier = Classifier()
    
    # Training data
    resume_texts = [
        "Python Java SQL database programming",
        "Accounting finance Excel budgeting",
        "Legal contracts law litigation"
    ]
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
    
    classifier.train_baseline(resume_texts, y_train)
    
    # Test probability predictions
    test_texts = ["Python programming"]
    probabilities = classifier.predict_proba(None, model_type="baseline", resume_texts=test_texts)
    
    assert probabilities.shape == (1, 3)  # 1 sample, 3 classes
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_predict_proba_proposed():
    """Test proposed model probability predictions."""
    classifier = Classifier()
    
    # Training data
    X_train = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0]
    ])
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
    
    classifier.train_proposed(X_train, y_train)
    
    # Test probability predictions
    X_test = np.array([[1, 1, 0, 0, 0, 0]])
    probabilities = classifier.predict_proba(X_test, model_type="proposed")
    
    assert probabilities.shape == (1, 3)  # 1 sample, 3 classes
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_validate_on_pdf_data():
    """Test validation on PDF-extracted features."""
    classifier = Classifier()
    
    # Train on CSV data
    X_train = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0]
    ])
    y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
    
    classifier.train_proposed(X_train, y_train)
    
    # Validate on PDF data
    X_pdf = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1]
    ])
    y_pdf = np.array(["IT", "ACCOUNTANT"])
    
    baseline_acc, proposed_acc = classifier.validate_on_pdf_data(X_pdf, y_pdf)
    
    assert baseline_acc is None  # Baseline not trained
    assert proposed_acc is not None
    assert 0.0 <= proposed_acc <= 1.0


def test_error_handling():
    """Test error handling for invalid inputs."""
    classifier = Classifier()
    
    # Test prediction without training
    with pytest.raises(ValueError, match="not trained"):
        classifier.predict(np.array([[1, 0, 0]]), model_type="proposed")
    
    # Test training with empty data
    with pytest.raises(ValueError, match="empty"):
        classifier.train_baseline([], np.array([]))
    
    # Test training with mismatched data
    with pytest.raises(ValueError, match="Mismatch"):
        classifier.train_proposed(np.array([[1, 0]]), np.array(["A", "B", "C"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
