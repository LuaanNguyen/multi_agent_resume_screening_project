#!/usr/bin/env python3
"""
Main execution script for the Smart Resume Screening System.

This script provides a command-line interface for processing resumes from CSV
and PDF sources, training ML models, running evaluations, and generating reports.
"""

import argparse
import logging
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

from src.models import ProcessorConfig, MLConfig, StructuredResume
from src.resume_processor import ResumeProcessor
from src.feature_generator import FeatureGenerator
from src.classifier import Classifier
from src.clustering_engine import ClusteringEngine
from src.association_miner import AssociationMiner
from src.evaluation_module import EvaluationModule
from src.logging_config import setup_logging


# Setup logging
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> tuple[ProcessorConfig, MLConfig]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Tuple of (ProcessorConfig, MLConfig)
    """
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        processor_config = ProcessorConfig(
            pdf_extractor=config_data.get('pdf_extractor', 'pdfplumber'),
            nlp_model=config_data.get('nlp_model', 'en_core_web_sm'),
            embedding_model=config_data.get('embedding_model', 'all-MiniLM-L6-v2'),
            fuzzy_threshold=config_data.get('fuzzy_threshold', 85),
            alias_dict_path=config_data.get('alias_dict_path', 'config/skill_aliases.json')
        )
        
        ml_config = MLConfig(
            n_clusters=config_data.get('n_clusters', 10),
            min_support=config_data.get('min_support', 0.1),
            min_confidence=config_data.get('min_confidence', 0.5),
            test_size=config_data.get('test_size', 0.2),
            random_state=config_data.get('random_state', 42)
        )
        
        logger.info(f"Configuration loaded from {config_path}")
        return processor_config, ml_config
        
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return ProcessorConfig(), MLConfig()


def save_structured_resumes(resumes: List[StructuredResume], output_dir: Path):
    """Save structured resumes as JSON files.
    
    Args:
        resumes: List of StructuredResume objects
        output_dir: Directory to save JSON files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for resume in resumes:
        output_file = output_dir / f"{resume.resume_id}.json"
        with open(output_file, 'w') as f:
            json.dump(resume.to_json(), f, indent=2)
    
    logger.info(f"Saved {len(resumes)} structured resumes to {output_dir}")


def process_csv_command(args):
    """Process resumes from CSV file.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== Starting CSV Processing ===")
    
    # Load configuration
    processor_config, _ = load_config(args.config)
    
    # Initialize processor
    processor = ResumeProcessor(processor_config)
    
    # Process CSV data
    logger.info(f"Processing CSV file: {args.csv_file}")
    structured_resumes = processor.process_csv_data(args.csv_file)
    
    # Save structured resumes
    output_dir = Path(args.output_dir) / "csv_structured"
    save_structured_resumes(structured_resumes, output_dir)
    
    logger.info(f"CSV processing complete: {len(structured_resumes)} resumes processed")


def process_pdf_command(args):
    """Process resumes from PDF archive directory.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== Starting PDF Processing ===")
    
    # Load configuration
    processor_config, _ = load_config(args.config)
    
    # Initialize processor
    processor = ResumeProcessor(processor_config)
    
    # Process PDF archive
    logger.info(f"Processing PDF archive: {args.pdf_dir}")
    resumes_by_category = processor.load_from_archive(args.pdf_dir)
    
    # Flatten and save structured resumes
    all_resumes = []
    for category, resumes in resumes_by_category.items():
        all_resumes.extend(resumes)
        logger.info(f"  {category}: {len(resumes)} resumes")
    
    output_dir = Path(args.output_dir) / "pdf_structured"
    save_structured_resumes(all_resumes, output_dir)
    
    logger.info(f"PDF processing complete: {len(all_resumes)} resumes processed")


def train_command(args):
    """Train ML models on CSV data.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== Starting Model Training ===")
    
    # Load configuration
    processor_config, ml_config = load_config(args.config)
    
    # Initialize components
    processor = ResumeProcessor(processor_config)
    feature_gen = FeatureGenerator()
    classifier = Classifier()
    
    # Load and process CSV data
    logger.info(f"Loading training data from: {args.csv_file}")
    structured_resumes = processor.process_csv_data(args.csv_file)
    
    # Generate features
    logger.info("Generating feature matrix...")
    X, vocabulary = feature_gen.generate_feature_matrix(structured_resumes)
    y = np.array([resume.job_category for resume in structured_resumes])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ml_config.test_size, random_state=ml_config.random_state
    )
    
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train baseline model
    logger.info("Training baseline model (TF-IDF + Logistic Regression)...")
    raw_texts = [resume.sections.raw_text for resume in structured_resumes]
    raw_train, raw_test = train_test_split(
        raw_texts, test_size=ml_config.test_size, random_state=ml_config.random_state
    )
    classifier.train_baseline(raw_train, y_train)
    
    # Train proposed model
    logger.info("Training proposed model (Skill Features + Random Forest)...")
    classifier.train_proposed(X_train, y_train)
    
    # Save models
    output_dir = Path(args.output_dir) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "classifier.pkl", 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(output_dir / "feature_generator.pkl", 'wb') as f:
        pickle.dump(feature_gen, f)
    
    with open(output_dir / "vocabulary.json", 'w') as f:
        json.dump(vocabulary, f, indent=2)
    
    logger.info(f"Models saved to {output_dir}")
    logger.info("Model training complete")


def evaluate_command(args):
    """Run evaluation on trained models.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== Starting Model Evaluation ===")
    
    # Load configuration
    processor_config, ml_config = load_config(args.config)
    
    # Load models
    models_dir = Path(args.output_dir) / "models"
    
    with open(models_dir / "classifier.pkl", 'rb') as f:
        classifier = pickle.load(f)
    
    with open(models_dir / "feature_generator.pkl", 'rb') as f:
        feature_gen = pickle.load(f)
    
    logger.info("Models loaded successfully")
    
    # Load test data
    processor = ResumeProcessor(processor_config)
    structured_resumes = processor.process_csv_data(args.csv_file)
    
    # Generate features
    X, _ = feature_gen.generate_feature_matrix(structured_resumes)
    y = np.array([resume.job_category for resume in structured_resumes])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ml_config.test_size, random_state=ml_config.random_state
    )
    
    # Also split raw texts for baseline model
    raw_texts = [resume.sections.raw_text for resume in structured_resumes]
    _, raw_test = train_test_split(
        raw_texts, test_size=ml_config.test_size, random_state=ml_config.random_state
    )
    
    # Get predictions
    logger.info("Generating predictions...")
    baseline_pred = classifier.predict(X_test, model_type="baseline", resume_texts=raw_test)
    proposed_pred = classifier.predict(X_test, model_type="proposed")
    
    # Evaluate
    evaluator = EvaluationModule()
    
    logger.info("Evaluating baseline model...")
    baseline_metrics = evaluator.evaluate_classification(y_test, baseline_pred)
    
    logger.info("Evaluating proposed model...")
    proposed_metrics = evaluator.evaluate_classification(y_test, proposed_pred)
    
    # Compare models
    logger.info("Comparing models...")
    comparison = evaluator.compare_models(baseline_metrics, proposed_metrics)
    
    # Fairness analysis
    logger.info("Analyzing fairness...")
    fairness_report = evaluator.analyze_fairness(y_test, proposed_pred, list(set(y)))
    
    # Generate and save report
    report = evaluator.generate_report(
        classification_metrics=proposed_metrics,
        comparison_report=comparison,
        fairness_report=fairness_report
    )
    
    output_dir = Path(args.output_dir) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_dir / 'evaluation_report.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nBaseline Model:")
    print(f"  Accuracy: {baseline_metrics.accuracy:.4f}")
    print(f"  Macro F1: {baseline_metrics.macro_f1:.4f}")
    print(f"\nProposed Model:")
    print(f"  Accuracy: {proposed_metrics.accuracy:.4f}")
    print(f"  Macro F1: {proposed_metrics.macro_f1:.4f}")
    print(f"\nImprovement:")
    print(f"  Accuracy: {comparison.accuracy_improvement:+.4f}")
    print(f"  Macro F1: {comparison.f1_improvement:+.4f}")
    print(f"\nFairness Analysis:")
    print(f"  Mean F1: {fairness_report.mean_f1:.4f}")
    print(f"  F1 Std Dev: {fairness_report.f1_std:.4f}")
    print(f"  Flagged Categories: {len(fairness_report.flagged_categories)}")
    if fairness_report.flagged_categories:
        print(f"    {', '.join(fairness_report.flagged_categories)}")
    print("="*60 + "\n")


def mine_command(args):
    """Run association mining on resume data.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== Starting Association Mining ===")
    
    # Load configuration
    processor_config, ml_config = load_config(args.config)
    
    # Initialize components
    processor = ResumeProcessor(processor_config)
    miner = AssociationMiner(
        min_support=ml_config.min_support,
        min_confidence=ml_config.min_confidence
    )
    
    # Load data
    logger.info(f"Loading data from: {args.csv_file}")
    structured_resumes = processor.process_csv_data(args.csv_file)
    
    # Mine associations
    logger.info("Mining skill associations...")
    rules = miner.mine_associations(structured_resumes)
    
    # Save rules
    output_dir = Path(args.output_dir) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rules_data = []
    for rule in rules:
        rules_data.append({
            "antecedents": list(rule.antecedents),
            "consequents": list(rule.consequents),
            "support": rule.support,
            "confidence": rule.confidence,
            "lift": rule.lift
        })
    
    with open(output_dir / "association_rules.json", 'w') as f:
        json.dump(rules_data, f, indent=2)
    
    logger.info(f"Association rules saved to {output_dir / 'association_rules.json'}")
    
    # Print top rules
    print("\n" + "="*60)
    print("TOP ASSOCIATION RULES (by lift)")
    print("="*60)
    
    sorted_rules = sorted(rules, key=lambda r: r.lift, reverse=True)[:10]
    for i, rule in enumerate(sorted_rules, 1):
        print(f"\n{i}. {set(rule.antecedents)} => {set(rule.consequents)}")
        print(f"   Support: {rule.support:.4f}, Confidence: {rule.confidence:.4f}, Lift: {rule.lift:.4f}")
    
    print("="*60 + "\n")


def validate_command(args):
    """Run cross-source validation between CSV and PDF data.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== Starting Cross-Source Validation ===")
    
    # Load configuration
    processor_config, ml_config = load_config(args.config)
    
    # Initialize components
    processor = ResumeProcessor(processor_config)
    evaluator = EvaluationModule()
    
    # Load CSV data
    logger.info(f"Loading CSV data from: {args.csv_file}")
    csv_resumes = processor.process_csv_data(args.csv_file)
    
    # Load PDF data
    logger.info(f"Loading PDF data from: {args.pdf_dir}")
    pdf_resumes_by_cat = processor.load_from_archive(args.pdf_dir)
    pdf_resumes = []
    for resumes in pdf_resumes_by_cat.values():
        pdf_resumes.extend(resumes)
    
    # Validate extraction pipeline
    logger.info("Validating PDF extraction against CSV ground truth...")
    extraction_report = evaluator.evaluate_extraction_pipeline(
        csv_resumes=csv_resumes,
        pdf_resumes=pdf_resumes
    )
    
    # Save validation report
    output_dir = Path(args.output_dir) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_data = {
        "extraction_validation": {
            "total_samples": extraction_report.total_samples,
            "text_similarity_mean": extraction_report.text_similarity_mean,
            "text_similarity_std": extraction_report.text_similarity_std,
            "skill_overlap_mean": extraction_report.skill_overlap_mean,
            "skill_overlap_std": extraction_report.skill_overlap_std,
            "extraction_accuracy": extraction_report.extraction_accuracy
        }
    }
    
    with open(output_dir / "validation_report.json", 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    logger.info(f"Validation report saved to {output_dir / 'validation_report.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-SOURCE VALIDATION SUMMARY")
    print("="*60)
    print(f"\nExtraction Pipeline Validation:")
    print(f"  Samples Compared: {extraction_report.total_samples}")
    print(f"  Text Similarity: {extraction_report.text_similarity_mean:.4f} ± {extraction_report.text_similarity_std:.4f}")
    print(f"  Skill Overlap: {extraction_report.skill_overlap_mean:.4f} ± {extraction_report.skill_overlap_std:.4f}")
    print(f"  Extraction Accuracy: {extraction_report.extraction_accuracy:.4f}")
    print("="*60 + "\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Smart Resume Screening System - Process resumes and train ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # process-csv command
    csv_parser = subparsers.add_parser(
        'process-csv',
        help='Extract and structure resumes from CSV file'
    )
    csv_parser.add_argument(
        '--csv-file',
        type=str,
        default='archive/Resume/Resume.csv',
        help='Path to CSV file (default: archive/Resume/Resume.csv)'
    )
    
    # process-pdf command
    pdf_parser = subparsers.add_parser(
        'process-pdf',
        help='Extract and structure resumes from PDF archive'
    )
    pdf_parser.add_argument(
        '--pdf-dir',
        type=str,
        default='archive/data/data',
        help='Path to PDF archive directory (default: archive/data/data)'
    )
    
    # train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train ML models on CSV data'
    )
    train_parser.add_argument(
        '--csv-file',
        type=str,
        default='archive/Resume/Resume.csv',
        help='Path to CSV file for training (default: archive/Resume/Resume.csv)'
    )
    
    # evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained models'
    )
    eval_parser.add_argument(
        '--csv-file',
        type=str,
        default='archive/Resume/Resume.csv',
        help='Path to CSV file for evaluation (default: archive/Resume/Resume.csv)'
    )
    
    # mine command
    mine_parser = subparsers.add_parser(
        'mine',
        help='Run association mining on resume data'
    )
    mine_parser.add_argument(
        '--csv-file',
        type=str,
        default='archive/Resume/Resume.csv',
        help='Path to CSV file (default: archive/Resume/Resume.csv)'
    )
    
    # validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Run cross-source validation between CSV and PDF data'
    )
    validate_parser.add_argument(
        '--csv-file',
        type=str,
        default='archive/Resume/Resume.csv',
        help='Path to CSV file (default: archive/Resume/Resume.csv)'
    )
    validate_parser.add_argument(
        '--pdf-dir',
        type=str,
        default='archive/data/data',
        help='Path to PDF archive directory (default: archive/data/data)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Execute command
    if args.command == 'process-csv':
        process_csv_command(args)
    elif args.command == 'process-pdf':
        process_pdf_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'mine':
        mine_command(args)
    elif args.command == 'validate':
        validate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
