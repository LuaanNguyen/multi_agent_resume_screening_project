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
from typing import List

from src.models import ProcessorConfig, MLConfig, StructuredResume
from src.logging_config import setup_logging


# Setup logging
logger = logging.getLogger(__name__)


# Heavy runtime dependencies are loaded lazily so `python main.py --help`
# works before the NLP/ML stack is installed and configured.
ResumeProcessor = None
FeatureGenerator = None
Classifier = None
AssociationMiner = None
EvaluationModule = None

_RUNTIME_IMPORTS = {
    "ResumeProcessor": ("src.resume_processor", "ResumeProcessor"),
    "FeatureGenerator": ("src.feature_generator", "FeatureGenerator"),
    "Classifier": ("src.classifier", "Classifier"),
    "AssociationMiner": ("src.association_miner", "AssociationMiner"),
    "EvaluationModule": ("src.evaluation_module", "EvaluationModule"),
}


def _get_runtime_dependency(name: str):
    """Load a runtime dependency only when a command actually needs it."""
    dependency = globals()[name]
    if dependency is None:
        module_name, attr_name = _RUNTIME_IMPORTS[name]
        module = __import__(module_name, fromlist=[attr_name])
        dependency = getattr(module, attr_name)
        globals()[name] = dependency
    return dependency


def _get_nested_value(config_data: dict, path: tuple[str, ...]):
    """Return a nested config value, or None if any path segment is missing."""
    current = config_data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _get_config_value(
    config_data: dict,
    flat_key: str,
    nested_path: tuple[str, ...],
    default,
):
    """Support both legacy flat config keys and the nested YAML structure."""
    if flat_key in config_data:
        return config_data[flat_key]

    nested_value = _get_nested_value(config_data, nested_path)
    return default if nested_value is None else nested_value


def _add_shared_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments that should work before or after the subcommand."""
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


def _build_extraction_validation_inputs(
    csv_resumes: List[StructuredResume],
    pdf_resumes: List[StructuredResume],
) -> tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """Align CSV/PDF resumes by ID for extraction validation."""
    csv_by_id = {resume.resume_id: resume for resume in csv_resumes}
    pdf_by_id = {resume.resume_id: resume for resume in pdf_resumes}

    common_ids = sorted(set(csv_by_id).intersection(pdf_by_id))
    if not common_ids:
        raise ValueError("No overlapping resume IDs found between CSV and PDF data")

    missing_pdf = len(csv_by_id) - len(common_ids)
    missing_csv = len(pdf_by_id) - len(common_ids)
    if missing_pdf or missing_csv:
        logger.warning(
            "Extraction validation will compare %d shared resumes "
            "(missing in PDF: %d, missing in CSV: %d)",
            len(common_ids),
            missing_pdf,
            missing_csv,
        )

    csv_texts = [csv_by_id[resume_id].sections.raw_text for resume_id in common_ids]
    pdf_texts = [pdf_by_id[resume_id].sections.raw_text for resume_id in common_ids]
    csv_skills = [csv_by_id[resume_id].normalized_skills for resume_id in common_ids]
    pdf_skills = [pdf_by_id[resume_id].normalized_skills for resume_id in common_ids]

    return csv_texts, pdf_texts, csv_skills, pdf_skills


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
            config_data = yaml.safe_load(f) or {}
        
        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a YAML mapping")
        
        processor_config = ProcessorConfig(
            pdf_extractor=_get_config_value(
                config_data,
                "pdf_extractor",
                ("processing", "pdf_extractor"),
                ProcessorConfig.pdf_extractor,
            ),
            nlp_model=_get_config_value(
                config_data,
                "nlp_model",
                ("nlp", "model"),
                ProcessorConfig.nlp_model,
            ),
            embedding_model=_get_config_value(
                config_data,
                "embedding_model",
                ("nlp", "embedding_model"),
                ProcessorConfig.embedding_model,
            ),
            fuzzy_threshold=_get_config_value(
                config_data,
                "fuzzy_threshold",
                ("skill_normalization", "fuzzy_threshold"),
                ProcessorConfig.fuzzy_threshold,
            ),
            alias_dict_path=_get_config_value(
                config_data,
                "alias_dict_path",
                ("skill_normalization", "alias_dict_path"),
                ProcessorConfig.alias_dict_path,
            ),
        )
        
        ml_config = MLConfig(
            n_clusters=_get_config_value(
                config_data,
                "n_clusters",
                ("ml", "clustering", "n_clusters"),
                MLConfig.n_clusters,
            ),
            min_support=_get_config_value(
                config_data,
                "min_support",
                ("ml", "association", "min_support"),
                MLConfig.min_support,
            ),
            min_confidence=_get_config_value(
                config_data,
                "min_confidence",
                ("ml", "association", "min_confidence"),
                MLConfig.min_confidence,
            ),
            test_size=_get_config_value(
                config_data,
                "test_size",
                ("ml", "classification", "test_size"),
                MLConfig.test_size,
            ),
            random_state=_get_config_value(
                config_data,
                "random_state",
                ("ml", "classification", "random_state"),
                MLConfig.random_state,
            ),
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
    resume_processor_cls = _get_runtime_dependency("ResumeProcessor")
    processor = resume_processor_cls(processor_config)
    
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
    resume_processor_cls = _get_runtime_dependency("ResumeProcessor")
    processor = resume_processor_cls(processor_config)
    
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
    import numpy as np
    
    # Load configuration
    processor_config, ml_config = load_config(args.config)
    
    # Initialize components
    resume_processor_cls = _get_runtime_dependency("ResumeProcessor")
    feature_generator_cls = _get_runtime_dependency("FeatureGenerator")
    classifier_cls = _get_runtime_dependency("Classifier")
    processor = resume_processor_cls(processor_config)
    feature_gen = feature_generator_cls()
    classifier = classifier_cls()
    
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
    import numpy as np
    
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
    resume_processor_cls = _get_runtime_dependency("ResumeProcessor")
    evaluation_module_cls = _get_runtime_dependency("EvaluationModule")
    processor = resume_processor_cls(processor_config)
    structured_resumes = processor.process_csv_data(args.csv_file)
    
    # Generate features
    X, _ = feature_gen.generate_feature_matrix(structured_resumes)
    y = np.array([resume.job_category for resume in structured_resumes])
    all_categories = sorted(set(y.tolist()))
    
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
    evaluator = evaluation_module_cls()
    
    logger.info("Evaluating baseline model...")
    baseline_metrics = evaluator.evaluate_classification(
        y_test, baseline_pred, all_categories
    )
    
    logger.info("Evaluating proposed model...")
    proposed_metrics = evaluator.evaluate_classification(
        y_test, proposed_pred, all_categories
    )
    
    # Compare models
    logger.info("Comparing models...")
    comparison = evaluator.compare_models(baseline_metrics, proposed_metrics)
    
    # Fairness analysis
    logger.info("Analyzing fairness...")
    fairness_report = evaluator.analyze_fairness(
        y_test, proposed_pred, all_categories
    )
    
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
    resume_processor_cls = _get_runtime_dependency("ResumeProcessor")
    association_miner_cls = _get_runtime_dependency("AssociationMiner")
    processor = resume_processor_cls(processor_config)
    miner = association_miner_cls(
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
    if not sorted_rules:
        print("\nNo association rules met the configured thresholds.")
    else:
        for i, rule in enumerate(sorted_rules, 1):
            print(f"\n{i}. {set(rule.antecedents)} => {set(rule.consequents)}")
            print(
                f"   Support: {rule.support:.4f}, "
                f"Confidence: {rule.confidence:.4f}, Lift: {rule.lift:.4f}"
            )
    
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
    resume_processor_cls = _get_runtime_dependency("ResumeProcessor")
    evaluation_module_cls = _get_runtime_dependency("EvaluationModule")
    processor = resume_processor_cls(processor_config)
    evaluator = evaluation_module_cls()
    
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
    csv_texts, pdf_texts, csv_skills, pdf_skills = _build_extraction_validation_inputs(
        csv_resumes, pdf_resumes
    )
    extraction_report = evaluator.evaluate_extraction_pipeline(
        csv_texts=csv_texts,
        pdf_texts=pdf_texts,
        csv_skills=csv_skills,
        pdf_skills=pdf_skills,
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
    print(f"  Text Similarity: {extraction_report.text_similarity_mean:.4f} +/- {extraction_report.text_similarity_std:.4f}")
    print(f"  Skill Overlap: {extraction_report.skill_overlap_mean:.4f} +/- {extraction_report.skill_overlap_std:.4f}")
    print(f"  Extraction Accuracy: {extraction_report.extraction_accuracy:.4f}")
    print("="*60 + "\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Smart Resume Screening System - Process resumes and train ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Shared arguments are added to both the root parser and subparsers so users
    # can place them either before or after the subcommand.
    _add_shared_cli_arguments(parser)
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # process-csv command
    csv_parser = subparsers.add_parser(
        'process-csv',
        help='Extract and structure resumes from CSV file'
    )
    _add_shared_cli_arguments(csv_parser)
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
    _add_shared_cli_arguments(pdf_parser)
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
    _add_shared_cli_arguments(train_parser)
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
    _add_shared_cli_arguments(eval_parser)
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
    _add_shared_cli_arguments(mine_parser)
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
    _add_shared_cli_arguments(validate_parser)
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
