"""
Demonstration script for ResumeProcessor with dual data source support.

This script demonstrates:
1. Processing resumes from CSV (fast, pre-extracted text)
2. Processing resumes from PDF archive (validates extraction pipeline)
3. Cross-validation between data sources
"""

from src.resume_processor import ResumeProcessor
from src.models import ProcessorConfig
import json


def main():
    print("=" * 70)
    print("Smart Resume Screening System - ResumeProcessor Demo")
    print("=" * 70)
    
    # Initialize processor
    print("\n1. Initializing ResumeProcessor...")
    config = ProcessorConfig(
        pdf_extractor="pdfplumber",
        nlp_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_threshold=85,
        alias_dict_path="config/skill_aliases.json"
    )
    processor = ResumeProcessor(config)
    print("   [PASS] Processor initialized with all components")
    
    # Demo 1: Process CSV data (primary training source)
    print("\n2. Processing resumes from CSV (primary data source)...")
    print("   Loading: archive/Resume/Resume.csv")
    csv_resumes = processor.process_csv_data("archive/Resume/Resume.csv")[:10]
    print(f"   [PASS] Processed {len(csv_resumes)} resumes from CSV")
    print(f"   Sample resume: {csv_resumes[0].resume_id}")
    print(f"   - Category: {csv_resumes[0].job_category}")
    print(f"   - Skills extracted: {len(csv_resumes[0].normalized_skills)}")
    print(f"   - Processing time: {csv_resumes[0].metadata.processing_time_ms}ms")
    
    # Demo 2: Process PDF archive (validates extraction)
    print("\n3. Processing resumes from PDF archive...")
    print("   Loading: archive/data/data/ACCOUNTANT/")
    pdf_resume = processor.process_resume(
        "archive/data/data/ACCOUNTANT/10554236.pdf",
        job_category="ACCOUNTANT"
    )
    print(f"   [PASS] Processed PDF resume: {pdf_resume.resume_id}")
    print(f"   - Category: {pdf_resume.job_category}")
    print(f"   - Skills extracted: {len(pdf_resume.normalized_skills)}")
    print(f"   - Processing time: {pdf_resume.metadata.processing_time_ms}ms")
    
    # Demo 3: Load from archive by category
    print("\n4. Loading resumes by job category from archive...")
    print("   Loading: archive/data/data/ (organized by category)")
    resumes_by_category = processor.load_from_archive("archive/data/data")
    print(f"   [PASS] Loaded resumes from {len(resumes_by_category)} categories")
    for category, resumes in list(resumes_by_category.items())[:3]:
        print(f"   - {category}: {len(resumes)} resumes")
    
    # Demo 4: Cross-validation
    print("\n5. Cross-validating PDF extraction against CSV ground truth...")
    validation_report = processor.cross_validate_data_sources(
        pdf_archive_path="archive/data/data",
        csv_path="archive/Resume/Resume.csv",
        max_samples=10
    )
    print(f"   [PASS] Validation complete:")
    print(f"   - Samples validated: {validation_report['total_validated']}")
    print(f"   - Success rate: {validation_report['success_rate']:.1%}")
    print(f"   - Avg similarity: {validation_report['average_similarity']:.3f}")
    print(f"   - Avg length ratio: {validation_report['average_length_ratio']:.3f}")
    
    # Demo 5: Export structured resume to JSON
    print("\n6. Exporting structured resume to JSON...")
    sample_resume = csv_resumes[0]
    json_output = sample_resume.to_json()
    print(f"   [PASS] Exported resume {sample_resume.resume_id}")
    print(f"   JSON keys: {list(json_output.keys())}")
    print(f"   Sample skills: {json_output['normalized_skills'][:5]}")
    
    print("\n" + "=" * 70)
    print("Demo complete! ResumeProcessor supports:")
    print("  [PASS] CSV data processing (fast, pre-extracted)")
    print("  [PASS] PDF archive processing (validates extraction)")
    print("  [PASS] Batch processing by job category")
    print("  [PASS] Cross-validation between data sources")
    print("  [PASS] Structured JSON export")
    print("=" * 70)


if __name__ == '__main__':
    main()
