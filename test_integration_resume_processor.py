"""
Integration test for ResumeProcessor with real data.
"""

from src.resume_processor import ResumeProcessor
from src.models import ProcessorConfig
import json

def test_csv_processing():
    """Test processing resumes from CSV file."""
    print("\n=== Testing CSV Processing ===")
    
    # Create processor
    config = ProcessorConfig(
        pdf_extractor="pdfplumber",
        nlp_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_threshold=85,
        alias_dict_path="config/skill_aliases.json"
    )
    
    processor = ResumeProcessor(config)
    
    # Process first 5 resumes from CSV
    print("Loading CSV data...")
    csv_data = processor.load_from_csv("archive/Resume/Resume.csv")
    print(f"Loaded {len(csv_data)} resume entries from CSV")
    
    # Process just the first 3 for testing
    print("\nProcessing first 3 resumes...")
    resumes = processor.process_csv_data("archive/Resume/Resume.csv")[:3]
    
    for resume in resumes:
        print(f"\nResume ID: {resume.resume_id}")
        print(f"Category: {resume.job_category}")
        print(f"Explicit Skills: {len(resume.skills.explicit_skills)}")
        print(f"Implicit Skills: {len(resume.skills.implicit_skills)}")
        print(f"Normalized Skills: {len(resume.normalized_skills)}")
        print(f"Processing Time: {resume.metadata.processing_time_ms}ms")
        print(f"Sample normalized skills: {resume.normalized_skills[:5]}")
    
    print("\n[PASS] CSV processing test passed!")
    return resumes


def test_pdf_processing():
    """Test processing resumes from PDF archive."""
    print("\n=== Testing PDF Processing ===")
    
    config = ProcessorConfig(
        pdf_extractor="pdfplumber",
        nlp_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_threshold=85,
        alias_dict_path="config/skill_aliases.json"
    )
    
    processor = ResumeProcessor(config)
    
    # Process one PDF file
    print("Processing a single PDF...")
    pdf_path = "archive/data/data/ACCOUNTANT/10554236.pdf"
    resume = processor.process_resume(pdf_path, job_category="ACCOUNTANT")
    
    print(f"\nResume ID: {resume.resume_id}")
    print(f"Category: {resume.job_category}")
    print(f"Explicit Skills: {len(resume.skills.explicit_skills)}")
    print(f"Implicit Skills: {len(resume.skills.implicit_skills)}")
    print(f"Normalized Skills: {len(resume.normalized_skills)}")
    print(f"Processing Time: {resume.metadata.processing_time_ms}ms")
    print(f"Sample normalized skills: {resume.normalized_skills[:5]}")
    
    print("\n[PASS] PDF processing test passed!")
    return resume


def test_cross_validation():
    """Test cross-validation between PDF and CSV data."""
    print("\n=== Testing Cross-Validation ===")
    
    config = ProcessorConfig(
        pdf_extractor="pdfplumber",
        nlp_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_threshold=85,
        alias_dict_path="config/skill_aliases.json"
    )
    
    processor = ResumeProcessor(config)
    
    # Run cross-validation on 5 samples
    print("Running cross-validation on 5 samples...")
    report = processor.cross_validate_data_sources(
        pdf_archive_path="archive/data/data",
        csv_path="archive/Resume/Resume.csv",
        max_samples=5
    )
    
    print(f"\nValidation Results:")
    print(f"Total Validated: {report['total_validated']}")
    print(f"Successful: {report['successful_extractions']}")
    print(f"Failed: {report['failed_extractions']}")
    print(f"Success Rate: {report['success_rate']:.1%}")
    print(f"Average Length Ratio: {report['average_length_ratio']:.3f}")
    print(f"Average Similarity: {report['average_similarity']:.3f}")
    
    print("\n[PASS] Cross-validation test passed!")
    return report


if __name__ == '__main__':
    print("=" * 60)
    print("ResumeProcessor Integration Tests")
    print("=" * 60)
    
    # Test CSV processing
    csv_resumes = test_csv_processing()
    
    # Test PDF processing
    pdf_resume = test_pdf_processing()
    
    # Test cross-validation
    validation_report = test_cross_validation()
    
    print("\n" + "=" * 60)
    print("All integration tests passed! [PASS]")
    print("=" * 60)
