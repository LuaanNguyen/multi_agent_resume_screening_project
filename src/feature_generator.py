"""
Feature Generator component for the Smart Resume Screening System.

This module provides the FeatureGenerator class that converts skills into
numerical feature vectors for machine learning, supporting both CSV and PDF
data sources.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

from src.models import StructuredResume
from src.resume_processor import ResumeProcessor

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Converts skills to numerical feature vectors for ML.
    
    This class builds a vocabulary of all unique normalized skills across the
    dataset and converts each resume's skills into a binary feature vector
    where feature[i] = 1 if vocabulary[i] is present in the resume, else 0.
    
    Supports both CSV data (Resume_str field) and PDF extraction pipeline.
    
    Attributes:
        vocabulary: List of unique normalized skills (built from dataset)
        vocabulary_index: Dictionary mapping skills to feature indices
    """
    
    def __init__(self):
        """Initialize feature generator."""
        self.vocabulary: List[str] = []
        self.vocabulary_index: Dict[str, int] = {}
        logger.info("FeatureGenerator initialized")
    
    def build_vocabulary(self, all_resumes: List[StructuredResume]) -> List[str]:
        """Create vocabulary of all unique normalized skills across dataset.
        
        This method collects all unique normalized skills from the provided
        resumes and builds a vocabulary that will be used for feature vector
        generation. The vocabulary is sorted alphabetically for consistency.
        
        Args:
            all_resumes: List of StructuredResume objects from any data source
            
        Returns:
            List of unique normalized skills (sorted alphabetically)
            
        Raises:
            ValueError: If no resumes provided or no skills found
        """
        if not all_resumes:
            raise ValueError("Cannot build vocabulary from empty resume list")
        
        logger.info(f"Building vocabulary from {len(all_resumes)} resumes")
        
        # Collect all unique skills
        unique_skills = set()
        
        for resume in all_resumes:
            # Add all normalized skills from this resume
            unique_skills.update(resume.normalized_skills)
        
        if not unique_skills:
            raise ValueError("No skills found in provided resumes")
        
        # Sort vocabulary alphabetically for consistency
        self.vocabulary = sorted(list(unique_skills))
        
        # Build index mapping for fast lookup
        self.vocabulary_index = {skill: idx for idx, skill in enumerate(self.vocabulary)}
        
        logger.info(f"Vocabulary built: {len(self.vocabulary)} unique skills")
        
        return self.vocabulary
    
    def generate_feature_vector(
        self,
        skills: List[str],
        vocabulary: List[str]
    ) -> np.ndarray:
        """Convert skills to binary feature vector based on vocabulary.
        
        Creates a binary vector where feature[i] = 1 if vocabulary[i] is present
        in the skills list, else 0.
        
        Args:
            skills: List of normalized skills from a resume
            vocabulary: Vocabulary list (if different from self.vocabulary)
            
        Returns:
            Binary numpy array of shape (len(vocabulary),)
        """
        # Use provided vocabulary or instance vocabulary
        vocab_to_use = vocabulary if vocabulary else self.vocabulary
        
        if not vocab_to_use:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        # Create binary vector
        feature_vector = np.zeros(len(vocab_to_use), dtype=np.int8)
        
        # Build index if using provided vocabulary
        if vocabulary:
            vocab_index = {skill: idx for idx, skill in enumerate(vocab_to_use)}
        else:
            vocab_index = self.vocabulary_index
        
        # Set features to 1 for present skills
        for skill in skills:
            if skill in vocab_index:
                feature_vector[vocab_index[skill]] = 1
        
        return feature_vector
    
    def generate_feature_matrix(
        self,
        all_resumes: List[StructuredResume]
    ) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix for all resumes.
        
        This method generates a feature matrix where each row represents a resume
        and each column represents a skill from the vocabulary. The vocabulary is
        built from the provided resumes if not already built.
        
        Args:
            all_resumes: List of StructuredResume objects
            
        Returns:
            Tuple of (feature_matrix, vocabulary) where:
                - feature_matrix: numpy array of shape (n_resumes, n_skills)
                - vocabulary: list of skill names corresponding to columns
        """
        if not all_resumes:
            raise ValueError("Cannot generate feature matrix from empty resume list")
        
        logger.info(f"Generating feature matrix for {len(all_resumes)} resumes")
        
        # Build vocabulary if not already built
        if not self.vocabulary:
            self.build_vocabulary(all_resumes)
        
        # Generate feature vectors for all resumes
        feature_vectors = []
        
        for resume in all_resumes:
            feature_vector = self.generate_feature_vector(
                skills=resume.normalized_skills,
                vocabulary=self.vocabulary
            )
            feature_vectors.append(feature_vector)
        
        # Stack vectors into matrix
        feature_matrix = np.vstack(feature_vectors)
        
        logger.info(
            f"Feature matrix generated: shape {feature_matrix.shape} "
            f"({feature_matrix.shape[0]} resumes x {feature_matrix.shape[1]} skills)"
        )
        
        return feature_matrix, self.vocabulary
    
    def load_csv_features(
        self,
        csv_path: str,
        processor: ResumeProcessor
    ) -> Tuple[np.ndarray, List[str], List[StructuredResume]]:
        """Generate features directly from CSV Resume_str data.
        
        This method processes CSV data and generates feature vectors using the
        Resume_str field directly, skipping PDF extraction.
        
        Args:
            csv_path: Path to CSV file with resume data
            processor: ResumeProcessor instance for processing CSV data
            
        Returns:
            Tuple of (feature_matrix, vocabulary, structured_resumes) where:
                - feature_matrix: numpy array of shape (n_resumes, n_skills)
                - vocabulary: list of skill names
                - structured_resumes: list of StructuredResume objects
        """
        logger.info(f"Loading features from CSV: {csv_path}")
        
        # Process CSV data
        structured_resumes = processor.process_csv_data(csv_path)
        
        if not structured_resumes:
            raise ValueError(f"No resumes processed from CSV: {csv_path}")
        
        # Generate feature matrix
        feature_matrix, vocabulary = self.generate_feature_matrix(structured_resumes)
        
        logger.info(
            f"CSV features loaded: {len(structured_resumes)} resumes, "
            f"{len(vocabulary)} skills"
        )
        
        return feature_matrix, vocabulary, structured_resumes
    
    def load_pdf_features(
        self,
        archive_path: str,
        processor: ResumeProcessor
    ) -> Tuple[np.ndarray, List[str], Dict[str, List[StructuredResume]]]:
        """Generate features from PDF extraction pipeline.
        
        This method processes PDF files from the archive directory and generates
        feature vectors from the extracted and normalized skills.
        
        Args:
            archive_path: Path to archive directory with job category subdirectories
            processor: ResumeProcessor instance for processing PDFs
            
        Returns:
            Tuple of (feature_matrix, vocabulary, resumes_by_category) where:
                - feature_matrix: numpy array of shape (n_resumes, n_skills)
                - vocabulary: list of skill names
                - resumes_by_category: dict mapping categories to resume lists
        """
        logger.info(f"Loading features from PDF archive: {archive_path}")
        
        # Load resumes from archive
        resumes_by_category = processor.load_from_archive(archive_path)
        
        # Flatten all resumes into single list
        all_resumes = []
        for category_resumes in resumes_by_category.values():
            all_resumes.extend(category_resumes)
        
        if not all_resumes:
            raise ValueError(f"No resumes processed from archive: {archive_path}")
        
        # Generate feature matrix
        feature_matrix, vocabulary = self.generate_feature_matrix(all_resumes)
        
        logger.info(
            f"PDF features loaded: {len(all_resumes)} resumes across "
            f"{len(resumes_by_category)} categories, {len(vocabulary)} skills"
        )
        
        return feature_matrix, vocabulary, resumes_by_category
