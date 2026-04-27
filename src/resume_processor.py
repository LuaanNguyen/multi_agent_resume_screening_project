"""
Resume Processor orchestrator for the Smart Resume Screening System.

This module provides the ResumeProcessor class that orchestrates the entire
resume processing pipeline with support for both CSV and PDF data sources.
"""

import logging
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.models import (
    StructuredResume,
    ResumeSections,
    SkillSet,
    ResumeMetadata,
    ProcessorConfig
)
from src.text_extractor import TextExtractor
from src.section_parser import SectionParser
from src.skill_extractor import SkillExtractor
from src.skill_normalizer import SkillNormalizer
from src.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)


class ResumeProcessor:
    """Orchestrates the complete resume processing pipeline.
    
    This class coordinates all components to process resumes from both PDF files
    and CSV data sources, generating structured resume data with extracted skills,
    normalized skills, and optional scoring metrics.
    
    Attributes:
        config: Processor configuration
        text_extractor: Component for PDF/text extraction
        section_parser: Component for section detection
        skill_extractor: Component for NLP-based skill extraction
        skill_normalizer: Component for skill normalization
        scoring_engine: Component for ATS and semantic scoring
    """
    
    def __init__(self, config: ProcessorConfig):
        """Initialize processor with all component dependencies.
        
        Args:
            config: ProcessorConfig with system configuration
        """
        self.config = config
        
        # Initialize all components
        logger.info("Initializing ResumeProcessor components")
        self.text_extractor = TextExtractor()
        self.section_parser = SectionParser()
        self.skill_extractor = SkillExtractor(nlp_model=config.nlp_model)
        
        # Load alias dictionary for skill normalizer
        alias_dict = self._load_alias_dict(config.alias_dict_path)
        self.skill_normalizer = SkillNormalizer(
            alias_dict=alias_dict,
            fuzzy_threshold=config.fuzzy_threshold
        )
        
        self._scoring_engine = None
        logger.info("ResumeProcessor initialization complete")

    @property
    def scoring_engine(self) -> ScoringEngine:
        """Load the scoring engine only when scoring is actually requested."""
        if self._scoring_engine is None:
            self._scoring_engine = ScoringEngine(
                embedding_model=self.config.embedding_model
            )
        return self._scoring_engine
    
    def _load_alias_dict(self, dict_path: str) -> Dict[str, str]:
        """Load skill alias dictionary from file.
        
        Args:
            dict_path: Path to alias dictionary JSON file
            
        Returns:
            Dictionary mapping skill variations to canonical forms
        """
        try:
            # Create a temporary normalizer just to load the dictionary
            temp_normalizer = SkillNormalizer(alias_dict={}, fuzzy_threshold=85)
            return temp_normalizer.load_alias_dictionary(dict_path)
        except FileNotFoundError:
            logger.warning(f"Alias dictionary not found at {dict_path}, using empty dict")
            return {}
    
    def process_resume(
        self,
        file_path: str,
        job_category: Optional[str] = None,
        resume_id: Optional[str] = None
    ) -> StructuredResume:
        """Process a single resume file (PDF or text).
        
        This method orchestrates the full pipeline:
        1. Extract text from PDF/text file
        2. Parse sections
        3. Extract skills (explicit and implicit)
        4. Normalize skills
        5. Generate StructuredResume
        
        Args:
            file_path: Path to resume file (PDF or text)
            job_category: Optional job category classification
            resume_id: Optional resume identifier (defaults to filename)
            
        Returns:
            StructuredResume with all fields populated
            
        Raises:
            Exception: If processing fails at any stage
        """
        start_time = time.time()
        
        try:
            # Generate resume ID from filename if not provided
            if resume_id is None:
                resume_id = Path(file_path).stem
            
            logger.info(f"Processing resume: {resume_id} from {file_path}")
            
            # Step 1: Extract text
            path = Path(file_path)
            if path.suffix.lower() == '.pdf':
                raw_text = self.text_extractor.extract_from_pdf(file_path)
            elif path.suffix.lower() == '.txt':
                raw_text = self.text_extractor.extract_from_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Clean the extracted text
            cleaned_text = self.text_extractor.clean_text(raw_text)
            
            # Step 2: Parse sections
            sections = self.section_parser.parse_sections(cleaned_text)
            
            # Step 3: Extract skills
            skill_set = self.skill_extractor.extract_all_skills(sections)
            
            # Step 4: Normalize skills
            all_skills = skill_set.all_skills()
            normalized_skills = self.skill_normalizer.normalize_skills(all_skills)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create metadata
            metadata = ResumeMetadata(
                file_path=file_path,
                processed_date=datetime.now().isoformat(),
                processing_time_ms=processing_time_ms
            )
            
            # Create StructuredResume
            structured_resume = StructuredResume(
                resume_id=resume_id,
                job_category=job_category or "UNKNOWN",
                sections=sections,
                skills=skill_set,
                normalized_skills=normalized_skills,
                scores=None,  # Scores calculated separately with job requirements
                metadata=metadata
            )
            
            logger.info(
                f"Successfully processed resume {resume_id}: "
                f"{len(normalized_skills)} skills in {processing_time_ms}ms"
            )
            
            return structured_resume
            
        except Exception as e:
            logger.error(f"Failed to process resume {file_path}: {e}")
            raise

    def process_batch(self, directory: str) -> List[StructuredResume]:
        """Process all resumes in a directory.
        
        This method processes all PDF and text files found in the specified
        directory (non-recursive).
        
        Args:
            directory: Path to directory containing resume files
            
        Returns:
            List of StructuredResume objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        logger.info(f"Processing batch from directory: {directory}")
        
        # Find all PDF and text files
        resume_files = []
        resume_files.extend(dir_path.glob("*.pdf"))
        resume_files.extend(dir_path.glob("*.txt"))
        
        logger.info(f"Found {len(resume_files)} resume files")
        
        # Process each resume
        structured_resumes = []
        for file_path in resume_files:
            try:
                structured_resume = self.process_resume(str(file_path))
                structured_resumes.append(structured_resume)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Continue processing other resumes
                continue
        
        logger.info(f"Successfully processed {len(structured_resumes)}/{len(resume_files)} resumes")
        return structured_resumes
    
    def load_from_archive(self, archive_path: str) -> Dict[str, List[StructuredResume]]:
        """Load resumes organized by job category folders.
        
        This method processes PDF files from an archive directory structure where
        resumes are organized into subdirectories by job category:
        archive_path/
            ACCOUNTANT/
                resume1.pdf
                resume2.pdf
            ADVOCATE/
                resume3.pdf
            ...
        
        Args:
            archive_path: Path to archive directory with job category subdirectories
            
        Returns:
            Dictionary mapping job categories to lists of StructuredResume objects
        """
        archive_dir = Path(archive_path)
        if not archive_dir.exists():
            raise FileNotFoundError(f"Archive directory not found: {archive_path}")
        
        logger.info(f"Loading resumes from archive: {archive_path}")
        
        # Dictionary to store resumes by category
        resumes_by_category: Dict[str, List[StructuredResume]] = {}
        
        # Iterate through subdirectories (job categories)
        for category_dir in archive_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            logger.info(f"Processing category: {category_name}")
            
            # Find all PDF files in this category
            pdf_files = list(category_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDFs in {category_name}")
            
            category_resumes = []
            for pdf_file in pdf_files:
                try:
                    # Process resume with category information
                    structured_resume = self.process_resume(
                        file_path=str(pdf_file),
                        job_category=category_name,
                        resume_id=pdf_file.stem
                    )
                    category_resumes.append(structured_resume)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {e}")
                    continue
            
            resumes_by_category[category_name] = category_resumes
            logger.info(
                f"Processed {len(category_resumes)}/{len(pdf_files)} resumes "
                f"for category {category_name}"
            )
        
        total_resumes = sum(len(resumes) for resumes in resumes_by_category.values())
        logger.info(
            f"Archive loading complete: {total_resumes} resumes across "
            f"{len(resumes_by_category)} categories"
        )
        
        return resumes_by_category
    
    def load_from_csv(self, csv_path: str) -> List[Dict[str, str]]:
        """Load resume data from CSV file.
        
        The CSV file should have columns: ID, Resume_str, Resume_html, Category
        
        Args:
            csv_path: Path to CSV file containing resume data
            
        Returns:
            List of dictionaries with resume data from CSV
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is missing required columns
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading resume data from CSV: {csv_path}")
        
        resume_data = []
        required_columns = {'ID', 'Resume_str', 'Resume_html', 'Category'}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate columns
            if not required_columns.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV missing required columns. Expected: {required_columns}, "
                    f"Found: {reader.fieldnames}"
                )
            
            # Load all rows
            for row in reader:
                resume_data.append(row)
        
        logger.info(f"Loaded {len(resume_data)} resume entries from CSV")
        return resume_data
    
    def process_csv_data(self, csv_path: str) -> List[StructuredResume]:
        """Process resume entries from CSV file.
        
        This method processes CSV resume data by using the Resume_str field directly,
        skipping PDF extraction. The pipeline includes:
        1. Load CSV data
        2. Parse sections from Resume_str
        3. Extract skills
        4. Normalize skills
        5. Generate StructuredResume
        
        Args:
            csv_path: Path to CSV file with resume data
            
        Returns:
            List of StructuredResume objects
        """
        # Load CSV data
        csv_data = self.load_from_csv(csv_path)
        
        logger.info(f"Processing {len(csv_data)} resumes from CSV")
        
        structured_resumes = []
        
        for idx, row in enumerate(csv_data):
            start_time = time.time()
            
            try:
                resume_id = row['ID']
                resume_text = row['Resume_str']
                job_category = row['Category']
                
                logger.debug(f"Processing CSV resume {idx + 1}/{len(csv_data)}: {resume_id}")
                
                # Clean the text
                cleaned_text = self.text_extractor.clean_text(resume_text)
                
                # Parse sections
                sections = self.section_parser.parse_sections(cleaned_text)
                
                # Extract skills
                skill_set = self.skill_extractor.extract_all_skills(sections)
                
                # Normalize skills
                all_skills = skill_set.all_skills()
                normalized_skills = self.skill_normalizer.normalize_skills(all_skills)
                
                # Calculate processing time
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Create metadata
                metadata = ResumeMetadata(
                    file_path=f"csv:{csv_path}:row_{idx}",
                    processed_date=datetime.now().isoformat(),
                    processing_time_ms=processing_time_ms
                )
                
                # Create StructuredResume
                structured_resume = StructuredResume(
                    resume_id=resume_id,
                    job_category=job_category,
                    sections=sections,
                    skills=skill_set,
                    normalized_skills=normalized_skills,
                    scores=None,
                    metadata=metadata
                )
                
                structured_resumes.append(structured_resume)
                
            except Exception as e:
                logger.error(f"Failed to process CSV row {idx}: {e}")
                continue
        
        logger.info(
            f"Successfully processed {len(structured_resumes)}/{len(csv_data)} "
            f"resumes from CSV"
        )
        
        return structured_resumes

    def validate_csv_extraction(
        self,
        pdf_path: str,
        csv_resume_str: str,
        resume_id: str
    ) -> Dict[str, any]:
        """Compare PDF extraction results against CSV Resume_str data.
        
        This method extracts text from a PDF file and compares it against the
        corresponding CSV Resume_str to validate extraction accuracy.
        
        Args:
            pdf_path: Path to PDF file
            csv_resume_str: Resume text from CSV Resume_str column
            resume_id: Resume identifier for logging
            
        Returns:
            Dictionary with validation metrics:
                - resume_id: Resume identifier
                - pdf_length: Character count from PDF extraction
                - csv_length: Character count from CSV
                - length_ratio: Ratio of PDF length to CSV length
                - text_similarity: Similarity score (0-1)
                - extraction_success: Boolean indicating if extraction is acceptable
        """
        try:
            # Extract text from PDF
            pdf_text = self.text_extractor.extract_from_pdf(pdf_path)
            pdf_cleaned = self.text_extractor.clean_text(pdf_text)
            
            # Clean CSV text
            csv_cleaned = self.text_extractor.clean_text(csv_resume_str)
            
            # Calculate metrics
            pdf_length = len(pdf_cleaned)
            csv_length = len(csv_cleaned)
            length_ratio = pdf_length / csv_length if csv_length > 0 else 0.0
            
            # Calculate text similarity using simple character overlap
            # (More sophisticated methods could use edit distance or embeddings)
            pdf_words = set(pdf_cleaned.lower().split())
            csv_words = set(csv_cleaned.lower().split())
            
            if len(csv_words) > 0:
                word_overlap = len(pdf_words.intersection(csv_words))
                text_similarity = word_overlap / len(csv_words)
            else:
                text_similarity = 0.0
            
            # Determine if extraction is acceptable
            # Criteria: length ratio between 0.7-1.3 and similarity > 0.6
            extraction_success = (
                0.7 <= length_ratio <= 1.3 and
                text_similarity >= 0.6
            )
            
            validation_result = {
                'resume_id': resume_id,
                'pdf_length': pdf_length,
                'csv_length': csv_length,
                'length_ratio': round(length_ratio, 3),
                'text_similarity': round(text_similarity, 3),
                'extraction_success': extraction_success
            }
            
            logger.debug(
                f"Validation for {resume_id}: "
                f"length_ratio={length_ratio:.3f}, "
                f"similarity={text_similarity:.3f}, "
                f"success={extraction_success}"
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed for {resume_id}: {e}")
            return {
                'resume_id': resume_id,
                'pdf_length': 0,
                'csv_length': len(csv_resume_str),
                'length_ratio': 0.0,
                'text_similarity': 0.0,
                'extraction_success': False,
                'error': str(e)
            }
    
    def cross_validate_data_sources(
        self,
        pdf_archive_path: str,
        csv_path: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, any]:
        """Identify discrepancies between PDF and CSV text.
        
        This method cross-validates PDF extraction against CSV ground truth data
        by matching resume IDs and comparing extracted text.
        
        Args:
            pdf_archive_path: Path to PDF archive with category subdirectories
            csv_path: Path to CSV file with resume data
            max_samples: Optional limit on number of resumes to validate
            
        Returns:
            Dictionary with validation report:
                - total_validated: Number of resumes validated
                - successful_extractions: Number of successful extractions
                - failed_extractions: Number of failed extractions
                - average_length_ratio: Average length ratio across all validations
                - average_similarity: Average text similarity
                - validation_details: List of individual validation results
        """
        logger.info("Starting cross-validation of PDF and CSV data sources")
        
        # Load CSV data
        csv_data = self.load_from_csv(csv_path)
        
        # Create lookup dictionary by resume ID
        csv_lookup = {row['ID']: row for row in csv_data}
        
        logger.info(f"Loaded {len(csv_lookup)} CSV entries for validation")
        
        # Find PDF files in archive
        archive_dir = Path(pdf_archive_path)
        pdf_files = []
        
        for category_dir in archive_dir.iterdir():
            if category_dir.is_dir():
                pdf_files.extend(category_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in archive")
        
        # Limit samples if specified
        if max_samples:
            pdf_files = pdf_files[:max_samples]
            logger.info(f"Limited validation to {max_samples} samples")
        
        # Validate each PDF against CSV
        validation_results = []
        
        for pdf_file in pdf_files:
            resume_id = pdf_file.stem
            
            # Check if this resume exists in CSV
            if resume_id not in csv_lookup:
                logger.debug(f"Resume {resume_id} not found in CSV, skipping")
                continue
            
            csv_row = csv_lookup[resume_id]
            
            # Validate extraction
            result = self.validate_csv_extraction(
                pdf_path=str(pdf_file),
                csv_resume_str=csv_row['Resume_str'],
                resume_id=resume_id
            )
            
            validation_results.append(result)
        
        # Calculate summary statistics
        total_validated = len(validation_results)
        successful = sum(1 for r in validation_results if r['extraction_success'])
        failed = total_validated - successful
        
        if total_validated > 0:
            avg_length_ratio = sum(r['length_ratio'] for r in validation_results) / total_validated
            avg_similarity = sum(r['text_similarity'] for r in validation_results) / total_validated
        else:
            avg_length_ratio = 0.0
            avg_similarity = 0.0
        
        # Generate report
        report = {
            'total_validated': total_validated,
            'successful_extractions': successful,
            'failed_extractions': failed,
            'success_rate': round(successful / total_validated, 3) if total_validated > 0 else 0.0,
            'average_length_ratio': round(avg_length_ratio, 3),
            'average_similarity': round(avg_similarity, 3),
            'validation_details': validation_results
        }
        
        logger.info(
            f"Cross-validation complete: {successful}/{total_validated} successful "
            f"({report['success_rate']:.1%} success rate)"
        )
        
        return report
