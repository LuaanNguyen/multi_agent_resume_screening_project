"""
Unit tests for the EvaluationModule component.

Tests cover classification metrics, clustering metrics, model comparison,
fairness analysis, extraction validation, and cross-source validation.
"""

import pytest
import numpy as np
from src.evaluation_module import (
    EvaluationModule,
    ClassificationMetrics,
    ClusteringMetrics,
    ComparisonReport,
    FairnessReport,
    ExtractionValidationReport,
    CrossSourceValidationReport
)


class TestEvaluationModuleInit:
    """Tests for EvaluationModule initialization."""
    
    def test_init_default_threshold(self):
        """Test initialization with default consistency threshold."""
        module = EvaluationModule()
        assert module.consistency_threshold == 0.05
    
    def test_init_custom_threshold(self):
        """Test initialization with custom consistency threshold."""
        module = EvaluationModule(consistency_threshold=0.1)
        assert module.consistency_threshold == 0.1


class TestEvaluateClassification:
    """Tests for classification evaluation."""
    
    def test_perfect_classification(self):
        """Test evaluation with perfect predictions."""
        module = EvaluationModule()
        y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        y_pred = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        
        metrics = module.evaluate_classification(y_true, y_pred)
        
        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert len(metrics.per_class_f1) == 3
        assert all(f1 == 1.0 for f1 in metrics.per_class_f1.values())
    
    def test_partial_classification(self):
        """Test evaluation with partial correct predictions."""
        module = EvaluationModule()
        y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        y_pred = np.array(['A', 'B', 'A', 'A', 'B', 'B'])
        
        metrics = module.evaluate_classification(y_true, y_pred)
        
        assert 0.0 < metrics.accuracy < 1.0
        assert 0.0 < metrics.macro_f1 < 1.0
        assert len(metrics.per_class_f1) == 3
    
    def test_with_class_names(self):
        """Test evaluation with explicit class names."""
        module = EvaluationModule()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        class_names = ['ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE']
        
        metrics = module.evaluate_classification(y_true, y_pred, class_names)
        
        assert 'ACCOUNTANT' in metrics.per_class_f1
        assert 'ADVOCATE' in metrics.per_class_f1
        assert 'AGRICULTURE' in metrics.per_class_f1
    
    def test_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        module = EvaluationModule()
        
        with pytest.raises(ValueError, match="empty arrays"):
            module.evaluate_classification(np.array([]), np.array([]))
    
    def test_mismatched_arrays_raises_error(self):
        """Test that mismatched arrays raise ValueError."""
        module = EvaluationModule()
        y_true = np.array(['A', 'B', 'C'])
        y_pred = np.array(['A', 'B'])
        
        with pytest.raises(ValueError, match="Mismatch"):
            module.evaluate_classification(y_true, y_pred)
    
    def test_zero_division_handling(self):
        """Test handling of classes with no predictions."""
        module = EvaluationModule()
        # Class C has no true positives or predictions
        y_true = np.array(['A', 'A', 'B', 'B', 'C'])
        y_pred = np.array(['A', 'B', 'A', 'B', 'A'])
        
        metrics = module.evaluate_classification(y_true, y_pred)
        
        # Should not raise error, F1 for C should be 0
        assert 'C' in metrics.per_class_f1
        assert metrics.per_class_f1['C'] == 0.0

    def test_single_sample_with_unseen_prediction_class(self):
        """Test evaluation when predictions include a class absent from y_true."""
        module = EvaluationModule()
        y_true = np.array(['A'])
        y_pred = np.array(['B'])

        metrics = module.evaluate_classification(y_true, y_pred)

        assert metrics.accuracy == 0.0
        assert metrics.macro_f1 == 0.0
        assert metrics.per_class_f1['A'] == 0.0
        assert metrics.per_class_f1['B'] == 0.0

    def test_full_class_vocabulary_can_exceed_observed_labels(self):
        """Test explicit class names for a tiny test split."""
        module = EvaluationModule()
        y_true = np.array(['ACCOUNTANT'])
        y_pred = np.array(['ENGINEERING'])
        class_names = ['ACCOUNTANT', 'ENGINEERING', 'SALES']

        metrics = module.evaluate_classification(y_true, y_pred, class_names)

        assert set(metrics.per_class_f1) == set(class_names)
        assert metrics.per_class_f1['ACCOUNTANT'] == 0.0
        assert metrics.per_class_f1['ENGINEERING'] == 0.0
        assert metrics.per_class_f1['SALES'] == 0.0


class TestEvaluateClustering:
    """Tests for clustering evaluation."""
    
    def test_good_clustering(self):
        """Test evaluation with well-separated clusters."""
        module = EvaluationModule()
        # Create well-separated clusters
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [10, 10], [10, 11], [11, 10]  # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        metrics = module.evaluate_clustering(X, labels)
        
        assert metrics.silhouette_score > 0.5  # Good separation
        assert metrics.n_clusters == 2
    
    def test_poor_clustering(self):
        """Test evaluation with overlapping clusters."""
        module = EvaluationModule()
        # Create overlapping clusters
        X = np.array([
            [0, 0], [1, 1], [2, 2],
            [0, 1], [1, 0], [1, 2]
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        metrics = module.evaluate_clustering(X, labels)
        
        # Silhouette score should be lower for overlapping clusters
        assert -1.0 <= metrics.silhouette_score <= 1.0
        assert metrics.n_clusters == 2
    
    def test_single_cluster_returns_zero(self):
        """Test that single cluster returns silhouette score of 0."""
        module = EvaluationModule()
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        metrics = module.evaluate_clustering(X, labels)
        
        assert metrics.silhouette_score == 0.0
        assert metrics.n_clusters == 1
    
    def test_empty_features_raises_error(self):
        """Test that empty feature matrix raises ValueError."""
        module = EvaluationModule()
        
        with pytest.raises(ValueError, match="empty feature matrix"):
            module.evaluate_clustering(np.array([]).reshape(0, 2), np.array([]))
    
    def test_mismatched_labels_raises_error(self):
        """Test that mismatched labels raise ValueError."""
        module = EvaluationModule()
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 1])
        
        with pytest.raises(ValueError, match="Mismatch"):
            module.evaluate_clustering(X, labels)
    
    def test_single_sample_returns_zero(self):
        """Test that single sample returns silhouette score of 0."""
        module = EvaluationModule()
        X = np.array([[0, 0]])
        labels = np.array([0])
        
        metrics = module.evaluate_clustering(X, labels)
        
        assert metrics.silhouette_score == 0.0


class TestCompareModels:
    """Tests for model comparison."""
    
    def test_proposed_better_than_baseline(self):
        """Test comparison when proposed model outperforms baseline."""
        module = EvaluationModule()
        
        baseline = ClassificationMetrics(
            accuracy=0.75,
            macro_f1=0.70,
            per_class_f1={'A': 0.7, 'B': 0.7}
        )
        proposed = ClassificationMetrics(
            accuracy=0.85,
            macro_f1=0.82,
            per_class_f1={'A': 0.8, 'B': 0.84}
        )
        
        report = module.compare_models(baseline, proposed)
        
        assert abs(report.accuracy_improvement - 0.10) < 1e-9
        assert abs(report.f1_improvement - 0.12) < 1e-9
        assert report.baseline_metrics == baseline
        assert report.proposed_metrics == proposed
    
    def test_baseline_better_than_proposed(self):
        """Test comparison when baseline outperforms proposed."""
        module = EvaluationModule()
        
        baseline = ClassificationMetrics(
            accuracy=0.85,
            macro_f1=0.82,
            per_class_f1={'A': 0.8, 'B': 0.84}
        )
        proposed = ClassificationMetrics(
            accuracy=0.75,
            macro_f1=0.70,
            per_class_f1={'A': 0.7, 'B': 0.7}
        )
        
        report = module.compare_models(baseline, proposed)
        
        assert abs(report.accuracy_improvement - (-0.10)) < 1e-9
        assert abs(report.f1_improvement - (-0.12)) < 1e-9
    
    def test_equal_performance(self):
        """Test comparison when models have equal performance."""
        module = EvaluationModule()
        
        baseline = ClassificationMetrics(
            accuracy=0.80,
            macro_f1=0.78,
            per_class_f1={'A': 0.78, 'B': 0.78}
        )
        proposed = ClassificationMetrics(
            accuracy=0.80,
            macro_f1=0.78,
            per_class_f1={'A': 0.78, 'B': 0.78}
        )
        
        report = module.compare_models(baseline, proposed)
        
        assert report.accuracy_improvement == 0.0
        assert report.f1_improvement == 0.0


class TestAnalyzeFairness:
    """Tests for fairness analysis."""
    
    def test_fair_performance_across_categories(self):
        """Test fairness analysis with consistent performance."""
        module = EvaluationModule()
        # All categories have similar performance
        y_true = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        y_pred = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        categories = ['A', 'B', 'C']
        
        report = module.analyze_fairness(y_true, y_pred, categories)
        
        assert report.mean_f1 == 1.0
        assert report.f1_variance == 0.0
        assert report.f1_std == 0.0
        assert len(report.flagged_categories) == 0
    
    def test_unfair_performance_flags_categories(self):
        """Test fairness analysis flags categories with poor performance."""
        module = EvaluationModule()
        # Category C has poor performance
        y_true = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        y_pred = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'A', 'B', 'A'])
        categories = ['A', 'B', 'C']
        
        report = module.analyze_fairness(y_true, y_pred, categories)
        
        assert report.mean_f1 > 0.0
        assert report.f1_variance > 0.0
        assert 'C' in report.flagged_categories
        assert report.per_category_f1['C'] < report.fairness_threshold
    
    def test_variance_calculation(self):
        """Test that variance is calculated correctly."""
        module = EvaluationModule()
        y_true = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        y_pred = np.array(['A', 'B', 'B', 'B', 'C', 'A'])
        categories = ['A', 'B', 'C']
        
        report = module.analyze_fairness(y_true, y_pred, categories)
        
        # Variance should be positive when performance differs
        assert report.f1_variance >= 0.0
        assert report.f1_std >= 0.0
        assert report.f1_std == np.sqrt(report.f1_variance)
    
    def test_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        module = EvaluationModule()
        
        with pytest.raises(ValueError, match="empty arrays"):
            module.analyze_fairness(np.array([]), np.array([]), [])
    
    def test_per_category_f1_scores(self):
        """Test that per-category F1 scores are calculated."""
        module = EvaluationModule()
        y_true = np.array(['A', 'A', 'B', 'B'])
        y_pred = np.array(['A', 'A', 'B', 'A'])
        categories = ['A', 'B']
        
        report = module.analyze_fairness(y_true, y_pred, categories)
        
        assert 'A' in report.per_category_f1
        assert 'B' in report.per_category_f1
        # A has 2 TP, 1 FP, 0 FN -> precision=2/3, recall=1.0, F1=0.8
        assert abs(report.per_category_f1['A'] - 0.8) < 1e-9
        # B has 1 TP, 0 FP, 1 FN -> precision=1.0, recall=0.5, F1=0.666...
        assert report.per_category_f1['B'] < 1.0


class TestEvaluateExtractionPipeline:
    """Tests for extraction pipeline validation."""
    
    def test_perfect_extraction(self):
        """Test validation with perfect extraction match."""
        module = EvaluationModule()
        csv_texts = ["Python Java SQL", "Machine Learning AI"]
        pdf_texts = ["Python Java SQL", "Machine Learning AI"]
        csv_skills = [["Python", "Java", "SQL"], ["Machine Learning", "AI"]]
        pdf_skills = [["Python", "Java", "SQL"], ["Machine Learning", "AI"]]
        
        report = module.evaluate_extraction_pipeline(
            csv_texts, pdf_texts, csv_skills, pdf_skills
        )
        
        assert report.total_samples == 2
        assert report.skill_overlap_mean == 1.0
        assert report.extraction_accuracy > 0.9
    
    def test_partial_extraction(self):
        """Test validation with partial extraction match."""
        module = EvaluationModule()
        csv_texts = ["Python Java SQL", "Machine Learning AI"]
        pdf_texts = ["Python Java", "Machine Learning"]
        csv_skills = [["Python", "Java", "SQL"], ["Machine Learning", "AI"]]
        pdf_skills = [["Python", "Java"], ["Machine Learning"]]
        
        report = module.evaluate_extraction_pipeline(
            csv_texts, pdf_texts, csv_skills, pdf_skills
        )
        
        assert report.total_samples == 2
        assert 0.0 < report.skill_overlap_mean < 1.0
        assert 0.0 < report.extraction_accuracy < 1.0
    
    def test_no_extraction(self):
        """Test validation with no extraction match."""
        module = EvaluationModule()
        csv_texts = ["Python Java SQL"]
        pdf_texts = [""]
        csv_skills = [["Python", "Java", "SQL"]]
        pdf_skills = [[]]
        
        report = module.evaluate_extraction_pipeline(
            csv_texts, pdf_texts, csv_skills, pdf_skills
        )
        
        assert report.total_samples == 1
        assert report.skill_overlap_mean == 0.0
    
    def test_empty_both_sources(self):
        """Test validation when both sources are empty."""
        module = EvaluationModule()
        csv_texts = [""]
        pdf_texts = [""]
        csv_skills = [[]]
        pdf_skills = [[]]
        
        report = module.evaluate_extraction_pipeline(
            csv_texts, pdf_texts, csv_skills, pdf_skills
        )
        
        # Empty on both sides should be considered perfect match
        assert report.skill_overlap_mean == 1.0
        assert report.text_similarity_mean == 1.0
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        module = EvaluationModule()
        
        with pytest.raises(ValueError, match="Mismatch"):
            module.evaluate_extraction_pipeline(
                ["text1", "text2"], ["text1"], [["skill"]], [["skill"]]
            )
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        module = EvaluationModule()
        
        with pytest.raises(ValueError, match="empty data"):
            module.evaluate_extraction_pipeline([], [], [], [])
    
    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        module = EvaluationModule()
        csv_texts = ["text1", "text2", "text3"]
        pdf_texts = ["text1", "text", "text3"]
        csv_skills = [["A"], ["B"], ["C"]]
        pdf_skills = [["A"], ["B", "C"], ["C"]]
        
        report = module.evaluate_extraction_pipeline(
            csv_texts, pdf_texts, csv_skills, pdf_skills
        )
        
        assert report.text_similarity_std >= 0.0
        assert report.skill_overlap_std >= 0.0


class TestCrossSourceValidation:
    """Tests for cross-source validation."""
    
    def test_consistent_performance(self):
        """Test validation with consistent performance across sources."""
        module = EvaluationModule(consistency_threshold=0.05)
        csv_y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        csv_y_pred = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        pdf_y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        pdf_y_pred = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        
        report = module.cross_source_validation(
            csv_y_true, csv_y_pred, pdf_y_true, pdf_y_pred
        )
        
        assert report.csv_accuracy == 1.0
        assert report.pdf_accuracy == 1.0
        assert report.accuracy_difference == 0.0
        assert report.consistent_performance is True
    
    def test_inconsistent_performance(self):
        """Test validation with inconsistent performance across sources."""
        module = EvaluationModule(consistency_threshold=0.05)
        csv_y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        csv_y_pred = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        pdf_y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        pdf_y_pred = np.array(['A', 'A', 'A', 'A', 'A', 'A'])
        
        report = module.cross_source_validation(
            csv_y_true, csv_y_pred, pdf_y_true, pdf_y_pred
        )
        
        assert report.csv_accuracy == 1.0
        assert report.pdf_accuracy < 1.0
        assert report.accuracy_difference > 0.05
        assert report.consistent_performance is False
    
    def test_small_difference_within_threshold(self):
        """Test that small differences are considered consistent."""
        module = EvaluationModule(consistency_threshold=0.1)
        csv_y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'])
        csv_y_pred = np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'])
        pdf_y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'])
        pdf_y_pred = np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'B'])
        
        report = module.cross_source_validation(
            csv_y_true, csv_y_pred, pdf_y_true, pdf_y_pred
        )
        
        # Difference is 0.1 (1 out of 10), which equals threshold
        assert abs(report.accuracy_difference - 0.1) < 1e-9
        assert report.consistent_performance is True
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        module = EvaluationModule()
        
        with pytest.raises(ValueError, match="empty data"):
            module.cross_source_validation(
                np.array([]), np.array([]), np.array([]), np.array([])
            )
    
    def test_f1_difference_calculation(self):
        """Test that F1 difference is calculated correctly."""
        module = EvaluationModule()
        csv_y_true = np.array(['A', 'B', 'A', 'B'])
        csv_y_pred = np.array(['A', 'B', 'A', 'B'])
        pdf_y_true = np.array(['A', 'B', 'A', 'B'])
        pdf_y_pred = np.array(['A', 'A', 'A', 'A'])
        
        report = module.cross_source_validation(
            csv_y_true, csv_y_pred, pdf_y_true, pdf_y_pred
        )
        
        assert report.csv_macro_f1 == 1.0
        assert report.pdf_macro_f1 < 1.0
        assert report.f1_difference > 0.0


class TestGenerateReport:
    """Tests for report generation."""
    
    def test_empty_report(self):
        """Test generating report with no metrics."""
        module = EvaluationModule()
        
        report = module.generate_report()
        
        assert report == {}
    
    def test_classification_only_report(self):
        """Test generating report with only classification metrics."""
        module = EvaluationModule()
        metrics = ClassificationMetrics(
            accuracy=0.85,
            macro_f1=0.82,
            per_class_f1={'A': 0.8, 'B': 0.84}
        )
        
        report = module.generate_report(classification_metrics=metrics)
        
        assert 'classification' in report
        assert report['classification']['accuracy'] == 0.85
        assert report['classification']['macro_f1'] == 0.82
    
    def test_clustering_only_report(self):
        """Test generating report with only clustering metrics."""
        module = EvaluationModule()
        metrics = ClusteringMetrics(silhouette_score=0.65, n_clusters=5)
        
        report = module.generate_report(clustering_metrics=metrics)
        
        assert 'clustering' in report
        assert report['clustering']['silhouette_score'] == 0.65
        assert report['clustering']['n_clusters'] == 5
    
    def test_full_report(self):
        """Test generating report with all metrics."""
        module = EvaluationModule()
        
        classification = ClassificationMetrics(0.85, 0.82, {'A': 0.8})
        clustering = ClusteringMetrics(0.65, 5)
        baseline = ClassificationMetrics(0.75, 0.70, {'A': 0.7})
        proposed = ClassificationMetrics(0.85, 0.82, {'A': 0.8})
        comparison = ComparisonReport(baseline, proposed, 0.10, 0.12)
        fairness = FairnessReport(
            {'A': 0.8, 'B': 0.6}, 0.7, 0.01, 0.1, ['B'], 0.6
        )
        extraction = ExtractionValidationReport(10, 0.9, 0.05, 0.85, 0.1, 0.875)
        cross_source = CrossSourceValidationReport(
            0.85, 0.83, 0.02, 0.82, 0.80, 0.02, True
        )
        
        report = module.generate_report(
            classification_metrics=classification,
            clustering_metrics=clustering,
            comparison_report=comparison,
            fairness_report=fairness,
            extraction_report=extraction,
            cross_source_report=cross_source
        )
        
        assert 'classification' in report
        assert 'clustering' in report
        assert 'model_comparison' in report
        assert 'fairness' in report
        assert 'extraction_validation' in report
        assert 'cross_source_validation' in report
    
    def test_comparison_report_structure(self):
        """Test that comparison report has correct structure."""
        module = EvaluationModule()
        
        baseline = ClassificationMetrics(0.75, 0.70, {'A': 0.7})
        proposed = ClassificationMetrics(0.85, 0.82, {'A': 0.8})
        comparison = ComparisonReport(baseline, proposed, 0.10, 0.12)
        
        report = module.generate_report(comparison_report=comparison)
        
        assert 'model_comparison' in report
        assert 'baseline' in report['model_comparison']
        assert 'proposed' in report['model_comparison']
        assert 'improvements' in report['model_comparison']
        assert report['model_comparison']['improvements']['accuracy'] == 0.10
        assert report['model_comparison']['improvements']['macro_f1'] == 0.12
