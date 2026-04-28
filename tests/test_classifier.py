"""
Comprehensive unit tests for the Classifier component.

This test suite covers:
- Baseline model training and prediction on CSV data
- Proposed model training and prediction on CSV-derived features
- Probability output format validation
- Handling of unknown categories
- Cross-validation between CSV-trained models and PDF-extracted features
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.classifier import Classifier


class TestClassifierInitialization:
    """Test Classifier initialization."""
    
    def test_classifier_initializes_with_none_models(self):
        """Test that Classifier initializes with None for all models."""
        classifier = Classifier()
        
        assert classifier.baseline_vectorizer is None
        assert classifier.baseline_model is None
        assert classifier.proposed_model is None
        assert classifier.classes_ is None


class TestBaselineModelTraining:
    """Test baseline model training on CSV data."""
    
    def test_train_baseline_with_csv_resume_texts(self):
        """Test baseline model training with CSV Resume_str data."""
        classifier = Classifier()
        
        # Simulate CSV Resume_str data
        resume_texts = [
            "Python Java SQL database programming software development",
            "Accounting finance Excel budgeting financial reporting tax",
            "Legal contracts law litigation court advocacy",
            "Python machine learning data science TensorFlow PyTorch",
            "Tax accounting financial statements audit compliance"
        ]
        
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
        
        # Train baseline model
        classifier.train_baseline(resume_texts, y_train)
        
        # Verify model components are initialized
        assert classifier.baseline_vectorizer is not None
        assert classifier.baseline_model is not None
        assert classifier.classes_ is not None
        assert len(classifier.classes_) == 3  # IT, ACCOUNTANT, ADVOCATE
        assert set(classifier.classes_) == {"IT", "ACCOUNTANT", "ADVOCATE"}
    
    def test_train_baseline_with_empty_texts_raises_error(self):
        """Test that training with empty resume texts raises ValueError."""
        classifier = Classifier()
        
        with pytest.raises(ValueError, match="empty resume texts"):
            classifier.train_baseline([], np.array([]))
    
    def test_train_baseline_with_mismatched_data_raises_error(self):
        """Test that mismatched texts and labels raise ValueError."""
        classifier = Classifier()
        
        resume_texts = ["Python programming", "Java development"]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])  # Mismatch
        
        with pytest.raises(ValueError, match="Mismatch"):
            classifier.train_baseline(resume_texts, y_train)
    
    def test_train_baseline_stores_class_labels(self):
        """Test that baseline training stores class labels correctly."""
        classifier = Classifier()
        
        resume_texts = [
            "Python programming",
            "Accounting finance",
            "Legal law"
        ]
        y_train = np.array(["TECH", "FINANCE", "LEGAL"])
        
        classifier.train_baseline(resume_texts, y_train)
        
        assert classifier.classes_ is not None
        assert len(classifier.classes_) == 3
        assert "TECH" in classifier.classes_
        assert "FINANCE" in classifier.classes_
        assert "LEGAL" in classifier.classes_


class TestProposedModelTraining:
    """Test proposed model training on CSV-derived features."""
    
    def test_train_proposed_with_csv_derived_skill_features(self):
        """Test proposed model training with binary skill features from CSV."""
        classifier = Classifier()
        
        # Simulate binary skill feature vectors derived from CSV data
        # 5 resumes, 10 skills
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
        
        # Verify model is initialized
        assert classifier.proposed_model is not None
        assert classifier.classes_ is not None
        assert len(classifier.classes_) == 3
    
    def test_train_proposed_with_empty_features_raises_error(self):
        """Test that training with empty feature matrix raises ValueError."""
        classifier = Classifier()
        
        X_train = np.array([]).reshape(0, 10)  # Empty matrix
        y_train = np.array([])
        
        with pytest.raises(ValueError, match="empty feature matrix"):
            classifier.train_proposed(X_train, y_train)
    
    def test_train_proposed_with_mismatched_data_raises_error(self):
        """Test that mismatched features and labels raise ValueError."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0, 0], [0, 1, 0]])
        y_train = np.array(["A", "B", "C"])  # Mismatch
        
        with pytest.raises(ValueError, match="Mismatch"):
            classifier.train_proposed(X_train, y_train)
    
    def test_train_proposed_uses_correct_hyperparameters(self):
        """Test that proposed model uses specified hyperparameters."""
        classifier = Classifier()
        
        X_train = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0]
        ])
        y_train = np.array(["A", "B", "C"])
        
        classifier.train_proposed(X_train, y_train)
        
        # Verify Random Forest hyperparameters
        assert classifier.proposed_model.n_estimators == 100
        assert classifier.proposed_model.max_depth == 20
        assert classifier.proposed_model.min_samples_split == 5
        assert classifier.proposed_model.random_state == 42


class TestBaselineModelPrediction:
    """Test baseline model predictions on CSV data."""
    
    def test_predict_baseline_returns_valid_labels(self):
        """Test that baseline predictions return valid job category labels."""
        classifier = Classifier()
        
        # Train model
        resume_texts = [
            "Python Java SQL database programming",
            "Accounting finance Excel budgeting",
            "Legal contracts law litigation"
        ]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_baseline(resume_texts, y_train)
        
        # Make predictions
        test_texts = ["Python programming SQL", "Finance accounting Excel"]
        predictions = classifier.predict(None, model_type="baseline", resume_texts=test_texts)
        
        assert len(predictions) == 2
        assert all(pred in ["IT", "ACCOUNTANT", "ADVOCATE"] for pred in predictions)
    
    def test_predict_baseline_without_training_raises_error(self):
        """Test that predicting without training raises ValueError."""
        classifier = Classifier()
        
        with pytest.raises(ValueError, match="not trained"):
            classifier.predict(None, model_type="baseline", resume_texts=["test"])
    
    def test_predict_baseline_without_texts_raises_error(self):
        """Test that baseline prediction without texts raises ValueError."""
        classifier = Classifier()
        
        # Train model with at least 2 classes
        resume_texts = ["Python programming", "Accounting finance"]
        y_train = np.array(["IT", "ACCOUNTANT"])
        classifier.train_baseline(resume_texts, y_train)
        
        # Try to predict without texts
        with pytest.raises(ValueError, match="resume_texts required"):
            classifier.predict(None, model_type="baseline", resume_texts=None)
    
    def test_predict_baseline_with_multiple_samples(self):
        """Test baseline predictions with multiple test samples."""
        classifier = Classifier()
        
        # Train with diverse data
        resume_texts = [
            "Python Java programming",
            "Accounting finance",
            "Legal law",
            "Machine learning AI",
            "Tax audit"
        ]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
        classifier.train_baseline(resume_texts, y_train)
        
        # Predict on multiple samples
        test_texts = [
            "Python programming",
            "Finance accounting",
            "Law litigation",
            "Java development"
        ]
        predictions = classifier.predict(None, model_type="baseline", resume_texts=test_texts)
        
        assert len(predictions) == 4
        assert all(isinstance(pred, (str, np.str_)) for pred in predictions)


class TestProposedModelPrediction:
    """Test proposed model predictions on CSV-derived features."""
    
    def test_predict_proposed_returns_valid_labels(self):
        """Test that proposed predictions return valid job category labels."""
        classifier = Classifier()
        
        # Train model
        X_train = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0]
        ])
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_proposed(X_train, y_train)
        
        # Make predictions
        X_test = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0]
        ])
        predictions = classifier.predict(X_test, model_type="proposed")
        
        assert len(predictions) == 2
        assert all(pred in ["IT", "ACCOUNTANT", "ADVOCATE"] for pred in predictions)
    
    def test_predict_proposed_without_training_raises_error(self):
        """Test that predicting without training raises ValueError."""
        classifier = Classifier()
        
        X_test = np.array([[1, 0, 0]])
        
        with pytest.raises(ValueError, match="not trained"):
            classifier.predict(X_test, model_type="proposed")
    
    def test_predict_proposed_with_empty_features_raises_error(self):
        """Test that prediction with empty features raises ValueError."""
        classifier = Classifier()
        
        # Train model
        X_train = np.array([[1, 0], [0, 1]])
        y_train = np.array(["A", "B"])
        classifier.train_proposed(X_train, y_train)
        
        # Try to predict with empty features
        with pytest.raises(ValueError, match="Feature matrix required"):
            classifier.predict(np.array([]).reshape(0, 2), model_type="proposed")
    
    def test_predict_proposed_with_invalid_model_type_raises_error(self):
        """Test that invalid model_type raises ValueError."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0]])
        y_train = np.array(["A"])
        classifier.train_proposed(X_train, y_train)
        
        with pytest.raises(ValueError, match="Invalid model_type"):
            classifier.predict(X_train, model_type="invalid")


class TestProbabilityOutputFormat:
    """Test probability output format for both models."""
    
    def test_predict_proba_baseline_returns_correct_shape(self):
        """Test that baseline probabilities have correct shape."""
        classifier = Classifier()
        
        # Train model
        resume_texts = [
            "Python Java SQL",
            "Accounting finance",
            "Legal law"
        ]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_baseline(resume_texts, y_train)
        
        # Get probabilities
        test_texts = ["Python programming", "Finance accounting"]
        probabilities = classifier.predict_proba(None, model_type="baseline", resume_texts=test_texts)
        
        # Verify shape: (n_samples, n_classes)
        assert probabilities.shape == (2, 3)
    
    def test_predict_proba_baseline_probabilities_sum_to_one(self):
        """Test that baseline probabilities sum to 1 for each sample."""
        classifier = Classifier()
        
        resume_texts = ["Python Java", "Accounting finance", "Legal law"]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_baseline(resume_texts, y_train)
        
        test_texts = ["Python", "Finance"]
        probabilities = classifier.predict_proba(None, model_type="baseline", resume_texts=test_texts)
        
        # Each row should sum to 1.0
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_predict_proba_baseline_values_in_valid_range(self):
        """Test that baseline probabilities are in [0, 1] range."""
        classifier = Classifier()
        
        resume_texts = ["Python Java", "Accounting finance", "Legal law"]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_baseline(resume_texts, y_train)
        
        test_texts = ["Python"]
        probabilities = classifier.predict_proba(None, model_type="baseline", resume_texts=test_texts)
        
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)
    
    def test_predict_proba_proposed_returns_correct_shape(self):
        """Test that proposed probabilities have correct shape."""
        classifier = Classifier()
        
        # Train model
        X_train = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0]
        ])
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_proposed(X_train, y_train)
        
        # Get probabilities
        X_test = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0]
        ])
        probabilities = classifier.predict_proba(X_test, model_type="proposed")
        
        # Verify shape: (n_samples, n_classes)
        assert probabilities.shape == (2, 3)
    
    def test_predict_proba_proposed_probabilities_sum_to_one(self):
        """Test that proposed probabilities sum to 1 for each sample."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_train = np.array(["A", "B", "C"])
        classifier.train_proposed(X_train, y_train)
        
        X_test = np.array([[1, 0, 0], [0, 1, 0]])
        probabilities = classifier.predict_proba(X_test, model_type="proposed")
        
        # Each row should sum to 1.0
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_predict_proba_proposed_values_in_valid_range(self):
        """Test that proposed probabilities are in [0, 1] range."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0], [0, 1]])
        y_train = np.array(["A", "B"])
        classifier.train_proposed(X_train, y_train)
        
        X_test = np.array([[1, 0]])
        probabilities = classifier.predict_proba(X_test, model_type="proposed")
        
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)
    
    def test_predict_proba_without_training_raises_error(self):
        """Test that predict_proba without training raises ValueError."""
        classifier = Classifier()
        
        with pytest.raises(ValueError, match="not trained"):
            classifier.predict_proba(np.array([[1, 0]]), model_type="proposed")


class TestHandlingUnknownCategories:
    """Test handling of unknown or unseen categories."""
    
    def test_predict_with_unseen_skill_patterns(self):
        """Test predictions on resumes with completely different skill patterns."""
        classifier = Classifier()
        
        # Train on specific skill patterns
        X_train = np.array([
            [1, 1, 0, 0, 0],  # Pattern A
            [0, 0, 1, 1, 0],  # Pattern B
            [0, 0, 0, 0, 1],  # Pattern C
        ])
        y_train = np.array(["CAT_A", "CAT_B", "CAT_C"])
        classifier.train_proposed(X_train, y_train)
        
        # Test with completely different pattern (all zeros)
        X_test = np.array([[0, 0, 0, 0, 0]])
        predictions = classifier.predict(X_test, model_type="proposed")
        
        # Should still return one of the trained categories
        assert len(predictions) == 1
        assert predictions[0] in ["CAT_A", "CAT_B", "CAT_C"]
    
    def test_predict_proba_with_unseen_patterns_returns_valid_probabilities(self):
        """Test that probabilities are valid even for unseen patterns."""
        classifier = Classifier()
        
        X_train = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        y_train = np.array(["A", "B", "C"])
        classifier.train_proposed(X_train, y_train)
        
        # Test with mixed pattern not seen in training
        X_test = np.array([[1, 1, 1]])
        probabilities = classifier.predict_proba(X_test, model_type="proposed")
        
        # Should still return valid probabilities
        assert probabilities.shape == (1, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)
    
    def test_baseline_with_completely_different_vocabulary(self):
        """Test baseline model with test texts having different vocabulary."""
        classifier = Classifier()
        
        # Train on specific vocabulary
        resume_texts = [
            "Python Java programming",
            "Accounting finance budgeting",
            "Legal law contracts"
        ]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE"])
        classifier.train_baseline(resume_texts, y_train)
        
        # Test with completely different vocabulary
        test_texts = ["Cooking culinary chef restaurant"]
        predictions = classifier.predict(None, model_type="baseline", resume_texts=test_texts)
        
        # Should still return a prediction (one of the trained categories)
        assert len(predictions) == 1
        assert predictions[0] in ["IT", "ACCOUNTANT", "ADVOCATE"]


class TestCrossValidationCSVvsPDF:
    """Test cross-validation between CSV-trained models and PDF-extracted features."""
    
    def test_validate_on_pdf_data_with_proposed_model(self):
        """Test validation of CSV-trained proposed model on PDF features."""
        classifier = Classifier()
        
        # Train on CSV-derived features
        X_train_csv = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0]
        ])
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
        classifier.train_proposed(X_train_csv, y_train)
        
        # Validate on PDF-extracted features
        X_pdf = np.array([
            [1, 1, 1, 0, 0, 0],  # Should be IT
            [0, 0, 0, 1, 1, 1],  # Should be ACCOUNTANT
        ])
        y_pdf = np.array(["IT", "ACCOUNTANT"])
        
        baseline_acc, proposed_acc = classifier.validate_on_pdf_data(X_pdf, y_pdf)
        
        # Baseline should be None (not trained)
        assert baseline_acc is None
        
        # Proposed accuracy should be valid
        assert proposed_acc is not None
        assert 0.0 <= proposed_acc <= 1.0
    
    def test_validate_on_pdf_data_with_both_models(self):
        """Test validation with both baseline and proposed models trained."""
        classifier = Classifier()
        
        # Train baseline on CSV texts
        resume_texts = [
            "Python Java SQL programming",
            "Accounting finance Excel",
            "Legal law contracts",
            "Machine learning AI",
            "Tax audit financial"
        ]
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
        classifier.train_baseline(resume_texts, y_train)
        
        # Train proposed on CSV-derived features
        X_train = np.array([
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1]
        ])
        classifier.train_proposed(X_train, y_train)
        
        # Validate on PDF data
        X_pdf = np.array([
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1]
        ])
        y_pdf = np.array(["IT", "ACCOUNTANT"])
        pdf_texts = [
            "Python Java SQL programming",
            "Accounting finance Excel"
        ]
        
        baseline_acc, proposed_acc = classifier.validate_on_pdf_data(
            X_pdf, y_pdf, pdf_resume_texts=pdf_texts
        )
        
        # Both accuracies should be valid
        assert baseline_acc is not None
        assert proposed_acc is not None
        assert 0.0 <= baseline_acc <= 1.0
        assert 0.0 <= proposed_acc <= 1.0
    
    def test_validate_on_pdf_data_without_proposed_model_raises_error(self):
        """Test that validation without proposed model raises ValueError."""
        classifier = Classifier()
        
        X_pdf = np.array([[1, 0, 0]])
        y_pdf = np.array(["IT"])
        
        with pytest.raises(ValueError, match="not trained"):
            classifier.validate_on_pdf_data(X_pdf, y_pdf)
    
    def test_validate_on_pdf_data_measures_extraction_quality(self):
        """Test that validation can detect extraction quality differences."""
        classifier = Classifier()
        
        # Train on clean CSV data
        X_train = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0]
        ])
        y_train = np.array(["IT", "ACCOUNTANT", "ADVOCATE", "IT", "ACCOUNTANT"])
        classifier.train_proposed(X_train, y_train)
        
        # Simulate perfect PDF extraction (same as training)
        X_pdf_perfect = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0]
        ])
        y_pdf = np.array(["IT", "ACCOUNTANT"])
        
        _, acc_perfect = classifier.validate_on_pdf_data(X_pdf_perfect, y_pdf)
        
        # Simulate noisy PDF extraction (missing some skills)
        X_pdf_noisy = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # Missing skills
            [0, 0, 0, 1, 0, 0, 0, 0]   # Missing skills
        ])
        
        _, acc_noisy = classifier.validate_on_pdf_data(X_pdf_noisy, y_pdf)
        
        # Perfect extraction should have better or equal accuracy
        assert acc_perfect >= acc_noisy
    
    def test_cross_validation_with_different_feature_dimensions(self):
        """Test that validation handles consistent feature dimensions."""
        classifier = Classifier()
        
        # Train with 6 features
        X_train = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1]
        ])
        y_train = np.array(["A", "B", "C"])
        classifier.train_proposed(X_train, y_train)
        
        # PDF data must have same 6 features
        X_pdf = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0]
        ])
        y_pdf = np.array(["A", "B"])
        
        _, proposed_acc = classifier.validate_on_pdf_data(X_pdf, y_pdf)
        
        assert proposed_acc is not None
        assert 0.0 <= proposed_acc <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_sample_training(self):
        """Test training with single sample (edge case)."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0, 0]])
        y_train = np.array(["SINGLE"])
        
        # Should train without error
        classifier.train_proposed(X_train, y_train)
        
        assert classifier.proposed_model is not None
        assert len(classifier.classes_) == 1
    
    def test_single_sample_prediction(self):
        """Test prediction with single test sample."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0], [0, 1]])
        y_train = np.array(["A", "B"])
        classifier.train_proposed(X_train, y_train)
        
        X_test = np.array([[1, 0]])
        predictions = classifier.predict(X_test, model_type="proposed")
        
        assert len(predictions) == 1
        assert predictions[0] in ["A", "B"]
    
    def test_all_zeros_feature_vector(self):
        """Test prediction with all-zeros feature vector."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_train = np.array(["A", "B", "C"])
        classifier.train_proposed(X_train, y_train)
        
        # Test with all zeros (no skills)
        X_test = np.array([[0, 0, 0]])
        predictions = classifier.predict(X_test, model_type="proposed")
        
        # Should still return a prediction
        assert len(predictions) == 1
        assert predictions[0] in ["A", "B", "C"]
    
    def test_all_ones_feature_vector(self):
        """Test prediction with all-ones feature vector."""
        classifier = Classifier()
        
        X_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_train = np.array(["A", "B", "C"])
        classifier.train_proposed(X_train, y_train)
        
        # Test with all ones (all skills)
        X_test = np.array([[1, 1, 1]])
        predictions = classifier.predict(X_test, model_type="proposed")
        
        # Should still return a prediction
        assert len(predictions) == 1
        assert predictions[0] in ["A", "B", "C"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
