"""
Classifier component for the Smart Resume Screening System.

This module provides the Classifier class that predicts job categories from
resume features using both baseline and proposed machine learning models.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models import StructuredResume

logger = logging.getLogger(__name__)


class Classifier:
    """Predicts job categories from resume features.
    
    This class implements two classification approaches:
    1. Baseline: TF-IDF vectors + Logistic Regression (trained on CSV Resume_str)
    2. Proposed: Binary skill features + Random Forest (trained on CSV-derived skills)
    
    The classifier is trained primarily on CSV data and can be validated on
    PDF-extracted features.
    
    Attributes:
        baseline_vectorizer: TfidfVectorizer for baseline model
        baseline_model: LogisticRegression classifier for baseline
        proposed_model: RandomForestClassifier for proposed approach
        classes_: Array of job category labels
    """
    
    def __init__(self):
        """Initialize classifier with baseline and proposed models."""
        # Baseline model components
        self.baseline_vectorizer: Optional[TfidfVectorizer] = None
        self.baseline_model: Optional[LogisticRegression] = None
        
        # Proposed model
        self.proposed_model: Optional[RandomForestClassifier] = None
        
        # Class labels
        self.classes_: Optional[np.ndarray] = None
        
        logger.info("Classifier initialized")
    
    def train_baseline(
        self,
        resume_texts: List[str],
        y_train: np.ndarray
    ):
        """Train TF-IDF + Logistic Regression baseline model.
        
        This method trains the baseline model using raw resume text (Resume_str
        from CSV) converted to TF-IDF features, then classified with Logistic
        Regression using L2 regularization.
        
        Args:
            resume_texts: List of raw resume text strings (from CSV Resume_str)
            y_train: Array of job category labels
            
        Raises:
            ValueError: If training data is empty or invalid
        """
        if not resume_texts or len(resume_texts) == 0:
            raise ValueError("Cannot train baseline model with empty resume texts")
        
        if len(resume_texts) != len(y_train):
            raise ValueError(
                f"Mismatch between resume texts ({len(resume_texts)}) "
                f"and labels ({len(y_train)})"
            )
        
        logger.info(f"Training baseline model on {len(resume_texts)} resumes")
        
        # Create TF-IDF vectorizer
        self.baseline_vectorizer = TfidfVectorizer()
        
        # Transform texts to TF-IDF features
        X_train_tfidf = self.baseline_vectorizer.fit_transform(resume_texts)
        
        logger.info(
            f"TF-IDF features: {X_train_tfidf.shape[0]} samples x "
            f"{X_train_tfidf.shape[1]} features"
        )
        
        # Train Logistic Regression with specified hyperparameters
        # Note: multi_class parameter removed in scikit-learn 1.5+
        # The solver (lbfgs) automatically handles multinomial classification
        self.baseline_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        self.baseline_model.fit(X_train_tfidf, y_train)
        
        # Store class labels
        self.classes_ = self.baseline_model.classes_
        
        logger.info(
            f"Baseline model trained: {len(self.classes_)} classes, "
            f"accuracy on training set: {self.baseline_model.score(X_train_tfidf, y_train):.4f}"
        )
    
    def train_proposed(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """Train skill features + Random Forest proposed model.
        
        This method trains the proposed model using binary skill feature vectors
        (derived from CSV data) with a Random Forest classifier.
        
        Args:
            X_train: Feature matrix of shape (n_samples, n_features) with binary
                    skill features from CSV-derived normalized skills
            y_train: Array of job category labels
            
        Raises:
            ValueError: If training data is empty or invalid
        """
        if X_train.shape[0] == 0:
            raise ValueError("Cannot train proposed model with empty feature matrix")
        
        if X_train.shape[0] != len(y_train):
            raise ValueError(
                f"Mismatch between features ({X_train.shape[0]}) "
                f"and labels ({len(y_train)})"
            )
        
        logger.info(
            f"Training proposed model on {X_train.shape[0]} resumes with "
            f"{X_train.shape[1]} skill features"
        )
        
        # Train Random Forest with specified hyperparameters
        self.proposed_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        
        self.proposed_model.fit(X_train, y_train)
        
        # Store class labels if not already set
        if self.classes_ is None:
            self.classes_ = self.proposed_model.classes_
        
        logger.info(
            f"Proposed model trained: {len(self.proposed_model.classes_)} classes, "
            f"accuracy on training set: {self.proposed_model.score(X_train, y_train):.4f}"
        )
    
    def predict(
        self,
        X: np.ndarray,
        model_type: str = "proposed",
        resume_texts: Optional[List[str]] = None
    ) -> np.ndarray:
        """Predict job categories for given features.
        
        Args:
            X: Feature matrix (for proposed model) or None (for baseline with texts)
            model_type: "baseline" or "proposed" (default: "proposed")
            resume_texts: List of resume texts (required for baseline model)
            
        Returns:
            Array of predicted job category labels
            
        Raises:
            ValueError: If model not trained or invalid inputs
        """
        if model_type == "baseline":
            if self.baseline_model is None or self.baseline_vectorizer is None:
                raise ValueError("Baseline model not trained. Call train_baseline() first.")
            
            if resume_texts is None:
                raise ValueError("resume_texts required for baseline model predictions")
            
            # Transform texts to TF-IDF features
            X_tfidf = self.baseline_vectorizer.transform(resume_texts)
            
            predictions = self.baseline_model.predict(X_tfidf)
            
            logger.debug(f"Baseline predictions: {len(predictions)} samples")
            
            return predictions
        
        elif model_type == "proposed":
            if self.proposed_model is None:
                raise ValueError("Proposed model not trained. Call train_proposed() first.")
            
            if X is None or X.shape[0] == 0:
                raise ValueError("Feature matrix required for proposed model predictions")
            
            predictions = self.proposed_model.predict(X)
            
            logger.debug(f"Proposed predictions: {len(predictions)} samples")
            
            return predictions
        
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Use 'baseline' or 'proposed'.")
    
    def predict_proba(
        self,
        X: np.ndarray,
        model_type: str = "proposed",
        resume_texts: Optional[List[str]] = None
    ) -> np.ndarray:
        """Return prediction confidence scores for each class.
        
        Args:
            X: Feature matrix (for proposed model) or None (for baseline with texts)
            model_type: "baseline" or "proposed" (default: "proposed")
            resume_texts: List of resume texts (required for baseline model)
            
        Returns:
            Array of shape (n_samples, n_classes) with probability scores
            
        Raises:
            ValueError: If model not trained or invalid inputs
        """
        if model_type == "baseline":
            if self.baseline_model is None or self.baseline_vectorizer is None:
                raise ValueError("Baseline model not trained. Call train_baseline() first.")
            
            if resume_texts is None:
                raise ValueError("resume_texts required for baseline model predictions")
            
            # Transform texts to TF-IDF features
            X_tfidf = self.baseline_vectorizer.transform(resume_texts)
            
            probabilities = self.baseline_model.predict_proba(X_tfidf)
            
            logger.debug(
                f"Baseline probabilities: {probabilities.shape[0]} samples x "
                f"{probabilities.shape[1]} classes"
            )
            
            return probabilities
        
        elif model_type == "proposed":
            if self.proposed_model is None:
                raise ValueError("Proposed model not trained. Call train_proposed() first.")
            
            if X is None or X.shape[0] == 0:
                raise ValueError("Feature matrix required for proposed model predictions")
            
            probabilities = self.proposed_model.predict_proba(X)
            
            logger.debug(
                f"Proposed probabilities: {probabilities.shape[0]} samples x "
                f"{probabilities.shape[1]} classes"
            )
            
            return probabilities
        
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Use 'baseline' or 'proposed'.")
    
    def validate_on_pdf_data(
        self,
        X_pdf: np.ndarray,
        y_pdf: np.ndarray,
        pdf_resume_texts: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """Test trained models on PDF-extracted features.
        
        This method validates both baseline and proposed models (trained on CSV
        data) against PDF-extracted features to assess extraction pipeline quality.
        
        Args:
            X_pdf: Feature matrix from PDF extraction pipeline
            y_pdf: True job category labels for PDF resumes
            pdf_resume_texts: Raw texts from PDF extraction (for baseline model)
            
        Returns:
            Tuple of (baseline_accuracy, proposed_accuracy)
            
        Raises:
            ValueError: If models not trained or invalid inputs
        """
        logger.info(f"Validating models on {X_pdf.shape[0]} PDF-extracted resumes")
        
        # Validate proposed model
        if self.proposed_model is None:
            raise ValueError("Proposed model not trained. Call train_proposed() first.")
        
        proposed_accuracy = self.proposed_model.score(X_pdf, y_pdf)
        
        logger.info(f"Proposed model accuracy on PDF data: {proposed_accuracy:.4f}")
        
        # Validate baseline model if texts provided
        baseline_accuracy = None
        
        if pdf_resume_texts is not None:
            if self.baseline_model is None or self.baseline_vectorizer is None:
                logger.warning("Baseline model not trained, skipping baseline validation")
            else:
                X_pdf_tfidf = self.baseline_vectorizer.transform(pdf_resume_texts)
                baseline_accuracy = self.baseline_model.score(X_pdf_tfidf, y_pdf)
                
                logger.info(f"Baseline model accuracy on PDF data: {baseline_accuracy:.4f}")
        
        return baseline_accuracy, proposed_accuracy
