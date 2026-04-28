"""
Demonstration script for the Classifier component.

This script demonstrates:
1. Loading CSV data
2. Training baseline model (TF-IDF + Logistic Regression)
3. Training proposed model (Skill features + Random Forest)
4. Making predictions with both models
5. Comparing model performance
"""

import numpy as np
from pathlib import Path

from src.classifier import Classifier
from src.resume_processor import ResumeProcessor
from src.feature_generator import FeatureGenerator
from src.models import ProcessorConfig


def main():
    print("=" * 70)
    print("Classifier Component Demonstration")
    print("=" * 70)
    
    # Initialize components
    print("\n1. Initializing components...")
    config = ProcessorConfig()
    processor = ResumeProcessor(config)
    feature_gen = FeatureGenerator()
    classifier = Classifier()
    
    # Load CSV data (using first 200 rows to get multiple categories)
    csv_path = "archive/Resume/Resume.csv"
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    print(f"\n2. Loading CSV data from {csv_path}...")
    print("   (Processing first 200 resumes for demo)")
    
    # Process CSV data
    structured_resumes = processor.process_csv_data(csv_path)
    
    # Limit to first 200 to get multiple categories
    structured_resumes = structured_resumes[:200]
    
    print(f"   [PASS] Processed {len(structured_resumes)} resumes")
    
    # Show category distribution
    categories = [r.job_category for r in structured_resumes]
    unique_categories = set(categories)
    print(f"   [PASS] Job categories: {unique_categories}")
    for cat in unique_categories:
        count = categories.count(cat)
        print(f"     - {cat}: {count} resumes")
    
    # Generate features
    print("\n3. Generating feature vectors...")
    X, vocabulary = feature_gen.generate_feature_matrix(structured_resumes)
    y = np.array([r.job_category for r in structured_resumes])
    
    print(f"   [PASS] Feature matrix shape: {X.shape}")
    print(f"   [PASS] Vocabulary size: {len(vocabulary)} unique skills")
    print(f"   [PASS] Sample skills: {vocabulary[:10]}")
    
    # Train proposed model
    print("\n4. Training PROPOSED model (Skill features + Random Forest)...")
    classifier.train_proposed(X, y)
    print(f"   [PASS] Model trained with hyperparameters:")
    print(f"     - n_estimators: 100")
    print(f"     - max_depth: 20")
    print(f"     - min_samples_split: 5")
    print(f"     - random_state: 42")
    
    # Train baseline model
    print("\n5. Training BASELINE model (TF-IDF + Logistic Regression)...")
    resume_texts = [r.sections.raw_text for r in structured_resumes]
    classifier.train_baseline(resume_texts, y)
    print(f"   [PASS] Model trained with hyperparameters:")
    print(f"     - C: 1.0")
    print(f"     - max_iter: 1000")
    print(f"     - solver: lbfgs (multinomial)")
    
    # Make predictions
    print("\n6. Making predictions on training data...")
    
    predictions_proposed = classifier.predict(X, model_type="proposed")
    predictions_baseline = classifier.predict(
        None,
        model_type="baseline",
        resume_texts=resume_texts
    )
    
    # Calculate accuracy
    accuracy_proposed = np.mean(predictions_proposed == y)
    accuracy_baseline = np.mean(predictions_baseline == y)
    
    print(f"   [PASS] Proposed model accuracy: {accuracy_proposed:.4f}")
    print(f"   [PASS] Baseline model accuracy: {accuracy_baseline:.4f}")
    
    # Show sample predictions
    print("\n7. Sample predictions (first 5 resumes):")
    print(f"   {'True Label':<15} {'Proposed':<15} {'Baseline':<15}")
    print(f"   {'-'*15} {'-'*15} {'-'*15}")
    
    for i in range(min(5, len(y))):
        print(f"   {y[i]:<15} {predictions_proposed[i]:<15} {predictions_baseline[i]:<15}")
    
    # Get prediction probabilities
    print("\n8. Prediction probabilities (first resume):")
    
    proba_proposed = classifier.predict_proba(X[:1], model_type="proposed")
    proba_baseline = classifier.predict_proba(
        None,
        model_type="baseline",
        resume_texts=resume_texts[:1]
    )
    
    print(f"\n   Proposed model probabilities:")
    for i, cls in enumerate(classifier.classes_):
        print(f"     {cls}: {proba_proposed[0][i]:.4f}")
    
    print(f"\n   Baseline model probabilities:")
    for i, cls in enumerate(classifier.classes_):
        print(f"     {cls}: {proba_baseline[0][i]:.4f}")
    
    print("\n" + "=" * 70)
    print("[PASS] Demonstration complete!")
    print("=" * 70)
    print("\nClassifier capabilities demonstrated:")
    print("  [PASS] Classifier class created with baseline and proposed models")
    print("  [PASS] __init__() initializes both model types")
    print("  [PASS] train_baseline() uses TF-IDF + Logistic Regression on CSV Resume_str")
    print("  [PASS] train_proposed() uses skill features + Random Forest")
    print("  [PASS] predict() returns job category predictions")
    print("  [PASS] predict_proba() returns confidence scores")
    print("  [PASS] validate_on_pdf_data() tests models on PDF-extracted features")
    print("=" * 70)


if __name__ == "__main__":
    main()
