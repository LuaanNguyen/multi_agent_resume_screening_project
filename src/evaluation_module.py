"""
Evaluation Module component for the Smart Resume Screening System.

This module provides the EvaluationModule class that measures system performance
and fairness across different data sources (CSV and PDF), including classification
metrics, clustering quality, model comparison, fairness analysis, and cross-source
validation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    silhouette_score,
    classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Metrics for classification model evaluation.
    
    Attributes:
        accuracy: Overall accuracy score (0-1)
        macro_f1: Macro-averaged F1 score (0-1)
        per_class_f1: Dictionary mapping class names to F1 scores
    """
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]


@dataclass
class ClusteringMetrics:
    """Metrics for clustering evaluation.
    
    Attributes:
        silhouette_score: Silhouette coefficient (-1 to 1)
        n_clusters: Number of clusters
    """
    silhouette_score: float
    n_clusters: int


@dataclass
class ComparisonReport:
    """Comparison between baseline and proposed models.
    
    Attributes:
        baseline_metrics: Metrics for baseline model
        proposed_metrics: Metrics for proposed model
        accuracy_improvement: Difference in accuracy (proposed - baseline)
        f1_improvement: Difference in macro F1 (proposed - baseline)
    """
    baseline_metrics: ClassificationMetrics
    proposed_metrics: ClassificationMetrics
    accuracy_improvement: float
    f1_improvement: float


@dataclass
class FairnessReport:
    """Fairness analysis across job categories.
    
    Attributes:
        per_category_f1: Dictionary mapping categories to F1 scores
        mean_f1: Mean F1 score across all categories
        f1_variance: Variance in F1 scores across categories
        f1_std: Standard deviation of F1 scores
        flagged_categories: List of categories with significantly lower performance
        fairness_threshold: Threshold used for flagging (mean - std_dev)
    """
    per_category_f1: Dict[str, float]
    mean_f1: float
    f1_variance: float
    f1_std: float
    flagged_categories: List[str]
    fairness_threshold: float


@dataclass
class ExtractionValidationReport:
    """Validation of PDF extraction accuracy against CSV ground truth.
    
    Attributes:
        total_samples: Number of samples compared
        text_similarity_mean: Mean text similarity score
        text_similarity_std: Standard deviation of text similarity
        skill_overlap_mean: Mean skill overlap percentage
        skill_overlap_std: Standard deviation of skill overlap
        extraction_accuracy: Overall extraction accuracy metric
    """
    total_samples: int
    text_similarity_mean: float
    text_similarity_std: float
    skill_overlap_mean: float
    skill_overlap_std: float
    extraction_accuracy: float


@dataclass
class CrossSourceValidationReport:
    """Comparison of model performance on CSV vs PDF-derived features.
    
    Attributes:
        csv_accuracy: Model accuracy on CSV data
        pdf_accuracy: Model accuracy on PDF data
        accuracy_difference: Absolute difference between CSV and PDF accuracy
        csv_macro_f1: Macro F1 on CSV data
        pdf_macro_f1: Macro F1 on PDF data
        f1_difference: Absolute difference between CSV and PDF F1
        consistent_performance: Whether performance is consistent across sources
    """
    csv_accuracy: float
    pdf_accuracy: float
    accuracy_difference: float
    csv_macro_f1: float
    pdf_macro_f1: float
    f1_difference: float
    consistent_performance: bool


class EvaluationModule:
    """Measures system performance and fairness across data sources.
    
    This class provides comprehensive evaluation capabilities including:
    - Classification performance metrics (accuracy, F1)
    - Clustering quality assessment (silhouette score)
    - Model comparison (baseline vs proposed)
    - Fairness analysis across job categories
    - PDF extraction validation against CSV ground truth
    - Cross-source performance validation (CSV vs PDF)
    
    Attributes:
        consistency_threshold: Maximum acceptable difference for cross-source validation
    """
    
    def __init__(self, consistency_threshold: float = 0.05):
        """Initialize evaluation module.
        
        Args:
            consistency_threshold: Maximum acceptable accuracy/F1 difference
                                  between CSV and PDF sources (default: 0.05)
        """
        self.consistency_threshold = consistency_threshold
        logger.info(
            f"EvaluationModule initialized with consistency_threshold={consistency_threshold}"
        )
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> ClassificationMetrics:
        """Calculate accuracy and macro-F1 score for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names for per-class metrics
            
        Returns:
            ClassificationMetrics with accuracy, macro F1, and per-class F1
            
        Raises:
            ValueError: If input arrays are empty or mismatched
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Cannot evaluate with empty arrays")
        
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Mismatch between true labels ({len(y_true)}) "
                f"and predictions ({len(y_pred)})"
            )
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Get unique classes from data if names not provided
        if class_names is None:
            # Important: Get labels from both true and pred to avoid IndexError
            # if the model predicts a class that isn't in y_true
            class_names = [str(c) for c in np.unique(np.concatenate([y_true, y_pred]))]
        
        # Calculate per-class F1 scores
        per_class_f1_scores = f1_score(
            y_true, y_pred, average=None, labels=np.unique(np.concatenate([y_true, y_pred])), zero_division=0
        )
        
        per_class_f1 = {
            class_names[i]: float(per_class_f1_scores[i])
            for i in range(len(per_class_f1_scores))
        }
        
        logger.info(
            f"Classification evaluation: accuracy={accuracy:.4f}, "
            f"macro_f1={macro_f1:.4f}"
        )
        
        return ClassificationMetrics(
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            per_class_f1=per_class_f1
        )
    
    def evaluate_clustering(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> ClusteringMetrics:
        """Calculate silhouette score for clustering quality.
        
        Args:
            X: Feature matrix used for clustering
            labels: Cluster labels assigned to each sample
            
        Returns:
            ClusteringMetrics with silhouette score and number of clusters
            
        Raises:
            ValueError: If inputs are invalid or insufficient samples
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate clustering with empty feature matrix")
        
        if len(labels) != X.shape[0]:
            raise ValueError(
                f"Mismatch between features ({X.shape[0]}) "
                f"and labels ({len(labels)})"
            )
        
        n_clusters = len(np.unique(labels))
        
        # Silhouette score requires at least 2 clusters and 2 samples
        if n_clusters < 2:
            logger.warning("Silhouette score requires at least 2 clusters")
            return ClusteringMetrics(silhouette_score=0.0, n_clusters=n_clusters)
        
        if X.shape[0] < 2:
            logger.warning("Silhouette score requires at least 2 samples")
            return ClusteringMetrics(silhouette_score=0.0, n_clusters=n_clusters)
        
        score = silhouette_score(X, labels)
        
        logger.info(
            f"Clustering evaluation: silhouette_score={score:.4f}, "
            f"n_clusters={n_clusters}"
        )
        
        return ClusteringMetrics(
            silhouette_score=float(score),
            n_clusters=n_clusters
        )
    
    def compare_models(
        self,
        baseline_metrics: ClassificationMetrics,
        proposed_metrics: ClassificationMetrics
    ) -> ComparisonReport:
        """Compare baseline vs proposed model performance.
        
        Args:
            baseline_metrics: Metrics from baseline model
            proposed_metrics: Metrics from proposed model
            
        Returns:
            ComparisonReport with both metrics and improvements
        """
        accuracy_improvement = (
            proposed_metrics.accuracy - baseline_metrics.accuracy
        )
        f1_improvement = (
            proposed_metrics.macro_f1 - baseline_metrics.macro_f1
        )
        
        logger.info(
            f"Model comparison: accuracy improvement={accuracy_improvement:+.4f}, "
            f"F1 improvement={f1_improvement:+.4f}"
        )
        
        return ComparisonReport(
            baseline_metrics=baseline_metrics,
            proposed_metrics=proposed_metrics,
            accuracy_improvement=float(accuracy_improvement),
            f1_improvement=float(f1_improvement)
        )
    
    def analyze_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        job_categories: List[str]
    ) -> FairnessReport:
        """Analyze performance across job categories for fairness.
        
        Calculates per-category F1 scores, variance, and flags categories
        with significantly lower performance (F1 < mean - std_dev).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            job_categories: List of job category names
            
        Returns:
            FairnessReport with per-category metrics and flagged categories
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Cannot analyze fairness with empty arrays")
        
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Mismatch between true labels ({len(y_true)}) "
                f"and predictions ({len(y_pred)})"
            )
        
        # Calculate per-category F1 scores
        per_class_f1_scores = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=job_categories
        )
        
        per_category_f1 = {
            job_categories[i]: float(per_class_f1_scores[i])
            for i in range(len(job_categories))
        }
        
        # Calculate statistics
        f1_values = list(per_category_f1.values())
        mean_f1 = float(np.mean(f1_values))
        f1_variance = float(np.var(f1_values))
        f1_std = float(np.std(f1_values))
        
        # Flag categories with F1 < (mean - std_dev)
        fairness_threshold = mean_f1 - f1_std
        flagged_categories = [
            category for category, f1 in per_category_f1.items()
            if f1 < fairness_threshold
        ]
        
        logger.info(
            f"Fairness analysis: mean_f1={mean_f1:.4f}, "
            f"variance={f1_variance:.4f}, "
            f"flagged_categories={len(flagged_categories)}/{len(job_categories)}"
        )
        
        if flagged_categories:
            logger.warning(
                f"Categories with lower performance: {', '.join(flagged_categories)}"
            )
        
        return FairnessReport(
            per_category_f1=per_category_f1,
            mean_f1=mean_f1,
            f1_variance=f1_variance,
            f1_std=f1_std,
            flagged_categories=flagged_categories,
            fairness_threshold=float(fairness_threshold)
        )
    
    def evaluate_extraction_pipeline(
        self,
        csv_texts: List[str],
        pdf_texts: List[str],
        csv_skills: List[List[str]],
        pdf_skills: List[List[str]]
    ) -> ExtractionValidationReport:
        """Validate PDF extraction accuracy against CSV ground truth.
        
        Compares PDF extraction results with CSV Resume_str data to assess
        extraction pipeline quality.
        
        Args:
            csv_texts: List of resume texts from CSV (ground truth)
            pdf_texts: List of resume texts from PDF extraction
            csv_skills: List of skill lists from CSV processing
            pdf_skills: List of skill lists from PDF extraction
            
        Returns:
            ExtractionValidationReport with similarity and overlap metrics
            
        Raises:
            ValueError: If inputs are mismatched or empty
        """
        if len(csv_texts) != len(pdf_texts):
            raise ValueError(
                f"Mismatch between CSV texts ({len(csv_texts)}) "
                f"and PDF texts ({len(pdf_texts)})"
            )
        
        if len(csv_skills) != len(pdf_skills):
            raise ValueError(
                f"Mismatch between CSV skills ({len(csv_skills)}) "
                f"and PDF skills ({len(pdf_skills)})"
            )
        
        if len(csv_texts) == 0:
            raise ValueError("Cannot validate with empty data")
        
        total_samples = len(csv_texts)
        
        # Calculate text similarity (simple character-based)
        text_similarities = []
        for csv_text, pdf_text in zip(csv_texts, pdf_texts):
            # Normalize texts
            csv_normalized = csv_text.lower().strip()
            pdf_normalized = pdf_text.lower().strip()
            
            # Calculate similarity as ratio of matching characters
            if len(csv_normalized) == 0 and len(pdf_normalized) == 0:
                similarity = 1.0
            elif len(csv_normalized) == 0 or len(pdf_normalized) == 0:
                similarity = 0.0
            else:
                # Simple length-based similarity
                min_len = min(len(csv_normalized), len(pdf_normalized))
                max_len = max(len(csv_normalized), len(pdf_normalized))
                similarity = min_len / max_len if max_len > 0 else 0.0
            
            text_similarities.append(similarity)
        
        # Calculate skill overlap
        skill_overlaps = []
        for csv_skill_list, pdf_skill_list in zip(csv_skills, pdf_skills):
            csv_set = set(s.lower() for s in csv_skill_list)
            pdf_set = set(s.lower() for s in pdf_skill_list)
            
            if len(csv_set) == 0 and len(pdf_set) == 0:
                overlap = 1.0
            elif len(csv_set) == 0 or len(pdf_set) == 0:
                overlap = 0.0
            else:
                intersection = len(csv_set.intersection(pdf_set))
                union = len(csv_set.union(pdf_set))
                overlap = intersection / union if union > 0 else 0.0
            
            skill_overlaps.append(overlap)
        
        # Calculate statistics
        text_similarity_mean = float(np.mean(text_similarities))
        text_similarity_std = float(np.std(text_similarities))
        skill_overlap_mean = float(np.mean(skill_overlaps))
        skill_overlap_std = float(np.std(skill_overlaps))
        
        # Overall extraction accuracy (weighted average)
        extraction_accuracy = (text_similarity_mean + skill_overlap_mean) / 2
        
        logger.info(
            f"Extraction validation: text_similarity={text_similarity_mean:.4f}, "
            f"skill_overlap={skill_overlap_mean:.4f}, "
            f"accuracy={extraction_accuracy:.4f}"
        )
        
        return ExtractionValidationReport(
            total_samples=total_samples,
            text_similarity_mean=text_similarity_mean,
            text_similarity_std=text_similarity_std,
            skill_overlap_mean=skill_overlap_mean,
            skill_overlap_std=skill_overlap_std,
            extraction_accuracy=float(extraction_accuracy)
        )
    
    def cross_source_validation(
        self,
        csv_y_true: np.ndarray,
        csv_y_pred: np.ndarray,
        pdf_y_true: np.ndarray,
        pdf_y_pred: np.ndarray
    ) -> CrossSourceValidationReport:
        """Compare model performance on CSV vs PDF-derived features.
        
        Validates that models trained on CSV data perform consistently when
        applied to PDF-extracted features.
        
        Args:
            csv_y_true: True labels for CSV data
            csv_y_pred: Predicted labels for CSV data
            pdf_y_true: True labels for PDF data
            pdf_y_pred: Predicted labels for PDF data
            
        Returns:
            CrossSourceValidationReport with performance comparison
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(csv_y_true) == 0 or len(pdf_y_true) == 0:
            raise ValueError("Cannot validate with empty data")
        
        # Calculate metrics for CSV data
        csv_accuracy = accuracy_score(csv_y_true, csv_y_pred)
        csv_macro_f1 = f1_score(csv_y_true, csv_y_pred, average='macro', zero_division=0)
        
        # Calculate metrics for PDF data
        pdf_accuracy = accuracy_score(pdf_y_true, pdf_y_pred)
        pdf_macro_f1 = f1_score(pdf_y_true, pdf_y_pred, average='macro', zero_division=0)
        
        # Calculate differences
        accuracy_difference = abs(csv_accuracy - pdf_accuracy)
        f1_difference = abs(csv_macro_f1 - pdf_macro_f1)
        
        # Check consistency
        consistent_performance = (
            accuracy_difference <= self.consistency_threshold and
            f1_difference <= self.consistency_threshold
        )
        
        logger.info(
            f"Cross-source validation: CSV accuracy={csv_accuracy:.4f}, "
            f"PDF accuracy={pdf_accuracy:.4f}, "
            f"difference={accuracy_difference:.4f}, "
            f"consistent={consistent_performance}"
        )
        
        if not consistent_performance:
            logger.warning(
                f"Performance inconsistency detected between CSV and PDF sources "
                f"(threshold={self.consistency_threshold})"
            )
        
        return CrossSourceValidationReport(
            csv_accuracy=float(csv_accuracy),
            pdf_accuracy=float(pdf_accuracy),
            accuracy_difference=float(accuracy_difference),
            csv_macro_f1=float(csv_macro_f1),
            pdf_macro_f1=float(pdf_macro_f1),
            f1_difference=float(f1_difference),
            consistent_performance=consistent_performance
        )
    
    def generate_report(
        self,
        classification_metrics: Optional[ClassificationMetrics] = None,
        clustering_metrics: Optional[ClusteringMetrics] = None,
        comparison_report: Optional[ComparisonReport] = None,
        fairness_report: Optional[FairnessReport] = None,
        extraction_report: Optional[ExtractionValidationReport] = None,
        cross_source_report: Optional[CrossSourceValidationReport] = None
    ) -> Dict[str, Any]:
        """Generate structured evaluation report.
        
        Args:
            classification_metrics: Optional classification metrics
            clustering_metrics: Optional clustering metrics
            comparison_report: Optional model comparison report
            fairness_report: Optional fairness analysis report
            extraction_report: Optional extraction validation report
            cross_source_report: Optional cross-source validation report
            
        Returns:
            Dictionary containing all provided metrics in structured format
        """
        report = {}
        
        if classification_metrics:
            report['classification'] = asdict(classification_metrics)
        
        if clustering_metrics:
            report['clustering'] = asdict(clustering_metrics)
        
        if comparison_report:
            report['model_comparison'] = {
                'baseline': asdict(comparison_report.baseline_metrics),
                'proposed': asdict(comparison_report.proposed_metrics),
                'improvements': {
                    'accuracy': comparison_report.accuracy_improvement,
                    'macro_f1': comparison_report.f1_improvement
                }
            }
        
        if fairness_report:
            report['fairness'] = asdict(fairness_report)
        
        if extraction_report:
            report['extraction_validation'] = asdict(extraction_report)
        
        if cross_source_report:
            report['cross_source_validation'] = asdict(cross_source_report)
        
        logger.info(f"Generated evaluation report with {len(report)} sections")
        
        return report
