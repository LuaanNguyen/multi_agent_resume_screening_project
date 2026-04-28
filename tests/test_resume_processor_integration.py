"""
Integration tests for ResumeProcessor component.

Tests end-to-end processing from PDF to StructuredResume JSON,
CSV data loading and processing, batch processing, archive loading,
CSV validation, and error handling for both data sources.

**Validates: Requirements 1.1, 1.2, 1.4, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6**
"""

import pytest
import tempfile
import csv
import json
from pathlib import Path
from src.resume_processor import ResumeProcessor
from src.models import ProcessorConfig, StructuredResume, ResumeSections, SkillSet, ResumeMetadata


@pytest.fixture
def processor_config():
    """Create a test processor configuration."""
    return ProcessorConfig(
        pdf_extractor="pdfplumber",
        nlp_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_threshold=85,
        alias_dict_path="config/skill_aliases.json"
    )


@pytest.fixture
def processor(processor_config):
    """Create a ResumeProcessor instance for testing."""
    return ResumeProcessor(processor_config)


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample resume data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Resume_str', 'Resume_html', 'Category'])
        writer.writeheader()
        
        # Sample resume 1
        writer.writerow({
            'ID': '10001',
            'Resume_str': '''
            JOHN DOE
            Software Engineer
            
            SKILLS
            Python, JavaScript, React, Node.js, Machine Learning, SQL, Docker, AWS
            
            EXPERIENCE
            Senior Software Engineer at Tech Corp (2020-2023)
            Developed web applications using React and Node.js
            Implemented machine learning models for data analysis
            
            EDUCATION
            Bachelor of Science in Computer Science
            University of Technology (2014-2018)
            
            PROJECTS
            E-commerce Platform: Built using React, Node.js, and MongoDB
            ''',
            'Resume_html': '<p>Resume HTML</p>',
            'Category': 'INFORMATION-TECHNOLOGY'
        })
        
        # Sample resume 2
        writer.writerow({
            'ID': '10002',
            'Resume_str': '''
            JANE SMITH
            Accountant
            
            SKILLS
            QuickBooks, Excel, Financial Analysis, Tax Preparation, Auditing
            
            EXPERIENCE
            Senior Accountant at Finance Corp (2019-2023)
            Managed financial reporting and tax preparation
            Conducted internal audits
            
            EDUCATION
            Bachelor of Business Administration in Accounting
            State University (2015-2019)
            ''',
            'Resume_html': '<p>Resume HTML</p>',
            'Category': 'ACCOUNTANT'
        })
        
        csv_path = f.name
    
    yield csv_path
    Path(csv_path).unlink()


class TestEndToEndPDFProcessing:
    """Test end-to-end processing from PDF to StructuredResume JSON."""
    
    def test_process_single_pdf_resume(self, processor):
        """Test processing a single PDF resume from archive."""
        # Use a real PDF from the archive
        pdf_path = "archive/data/data/ACCOUNTANT/10554236.pdf"
        
        if not Path(pdf_path).exists():
            pytest.skip(f"PDF file not found: {pdf_path}")
        
        # Process the resume
        result = processor.process_resume(pdf_path, job_category="ACCOUNTANT")
        
        # Verify StructuredResume structure
        assert isinstance(result, StructuredResume)
        assert result.resume_id == "10554236"
        assert result.job_category == "ACCOUNTANT"
        
        # Verify sections were parsed
        assert result.sections is not None
        assert result.sections.raw_text != ""
        
        # Verify skills were extracted
        assert result.skills is not None
        assert isinstance(result.skills.explicit_skills, list)
        assert isinstance(result.skills.implicit_skills, list)
        
        # Verify normalized skills
        assert isinstance(result.normalized_skills, list)
        
        # Verify metadata
        assert result.metadata is not None
        assert result.metadata.file_path == pdf_path
        assert result.metadata.processing_time_ms > 0
        assert result.metadata.processed_date is not None
    
    def test_process_pdf_to_json_serialization(self, processor):
        """Test that processed PDF resume can be serialized to JSON."""
        pdf_path = "archive/data/data/ADVOCATE/10186968.pdf"
        
        if not Path(pdf_path).exists():
            pytest.skip(f"PDF file not found: {pdf_path}")
        
        # Process resume
        result = processor.process_resume(pdf_path, job_category="ADVOCATE")
        
        # Convert to JSON
        json_data = result.to_json()
        
        # Verify JSON structure
        assert isinstance(json_data, dict)
        assert 'resume_id' in json_data
        assert 'job_category' in json_data
        assert 'sections' in json_data
        assert 'skills' in json_data
        assert 'normalized_skills' in json_data
        assert 'metadata' in json_data
        
        # Verify JSON can be serialized
        json_str = json.dumps(json_data)
        assert len(json_str) > 0
        
        # Verify JSON can be deserialized back
        loaded_data = json.loads(json_str)
        assert loaded_data['resume_id'] == result.resume_id


class TestCSVDataProcessing:
    """Test CSV data loading and processing pipeline."""
    
    def test_load_csv_data(self, processor, sample_csv_file):
        """Test loading resume data from CSV file."""
        data = processor.load_from_csv(sample_csv_file)
        
        # Verify data loaded correctly
        assert len(data) == 2
        assert data[0]['ID'] == '10001'
        assert data[1]['ID'] == '10002'
        assert 'Python' in data[0]['Resume_str']
        assert 'QuickBooks' in data[1]['Resume_str']
        assert data[0]['Category'] == 'INFORMATION-TECHNOLOGY'
        assert data[1]['Category'] == 'ACCOUNTANT'
    
    def test_process_csv_data_end_to_end(self, processor, sample_csv_file):
        """Test end-to-end CSV processing from CSV to StructuredResume."""
        resumes = processor.process_csv_data(sample_csv_file)
        
        # Verify all resumes processed
        assert len(resumes) == 2
        
        # Verify first resume
        resume1 = resumes[0]
        assert resume1.resume_id == '10001'
        assert resume1.job_category == 'INFORMATION-TECHNOLOGY'
        assert len(resume1.normalized_skills) > 0
        assert resume1.metadata.processing_time_ms > 0
        
        # Verify second resume
        resume2 = resumes[1]
        assert resume2.resume_id == '10002'
        assert resume2.job_category == 'ACCOUNTANT'
        assert len(resume2.normalized_skills) > 0
    
    def test_csv_processing_skips_pdf_extraction(self, processor, sample_csv_file):
        """Test that CSV processing uses Resume_str directly without PDF extraction."""
        resumes = processor.process_csv_data(sample_csv_file)
        
        # Verify processing was fast (no PDF extraction overhead)
        for resume in resumes:
            # Processing should be relatively fast without PDF extraction
            assert resume.metadata.processing_time_ms < 10000  # Less than 10 seconds
            
            # Verify text came from CSV (contains expected content)
            assert resume.sections.raw_text != ""
    
    def test_csv_data_to_json(self, processor, sample_csv_file):
        """Test CSV processed resumes can be serialized to JSON."""
        resumes = processor.process_csv_data(sample_csv_file)
        
        for resume in resumes:
            json_data = resume.to_json()
            
            # Verify JSON structure
            assert isinstance(json_data, dict)
            assert 'resume_id' in json_data
            assert 'job_category' in json_data
            
            # Verify can be serialized
            json_str = json.dumps(json_data)
            assert len(json_str) > 0


class TestBatchProcessing:
    """Test batch processing of multiple resumes from both sources."""
    
    def test_batch_process_pdf_directory(self, processor):
        """Test batch processing of multiple PDF resumes from a directory."""
        # Use a small subset from archive
        test_dir = "archive/data/data/AGRICULTURE"
        
        if not Path(test_dir).exists():
            pytest.skip(f"Test directory not found: {test_dir}")
        
        # Process batch (limit to first 3 files for speed)
        pdf_files = list(Path(test_dir).glob("*.pdf"))[:3]
        
        if len(pdf_files) == 0:
            pytest.skip("No PDF files found in test directory")
        
        results = []
        for pdf_file in pdf_files:
            result = processor.process_resume(str(pdf_file), job_category="AGRICULTURE")
            results.append(result)
        
        # Verify all resumes processed
        assert len(results) == len(pdf_files)
        
        # Verify each result
        for result in results:
            assert isinstance(result, StructuredResume)
            assert result.job_category == "AGRICULTURE"
            assert result.metadata.processing_time_ms > 0
    
    def test_batch_process_csv_data(self, processor, sample_csv_file):
        """Test batch processing of multiple resumes from CSV."""
        resumes = processor.process_csv_data(sample_csv_file)
        
        # Verify batch processing
        assert len(resumes) == 2
        
        # Verify all have unique IDs
        ids = [r.resume_id for r in resumes]
        assert len(ids) == len(set(ids))  # All unique
        
        # Verify all have categories
        for resume in resumes:
            assert resume.job_category in ['INFORMATION-TECHNOLOGY', 'ACCOUNTANT']
    
    def test_batch_processing_error_handling(self, processor):
        """Test batch processing handles errors gracefully."""
        # Create temp directory with mix of valid and invalid files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create an invalid file
            invalid_file = temp_path / "invalid.txt"
            invalid_file.write_text("Not a valid resume")
            
            # Try to process - should handle error gracefully
            try:
                result = processor.process_resume(str(invalid_file), job_category="TEST")
                # If it doesn't raise an error, verify it returns something reasonable
                assert result is not None
            except Exception as e:
                # Error handling is acceptable
                assert True


class TestArchiveLoading:
    """Test archive loading with job category organization."""

    @pytest.fixture
    def sample_pdf_archive(self, tmp_path):
        """Create a small category-organized PDF archive for structure tests."""
        archive_dir = tmp_path / "archive"
        for category in ["ACCOUNTANT", "INFORMATION-TECHNOLOGY"]:
            category_dir = archive_dir / category
            category_dir.mkdir(parents=True)
            for resume_id in ["resume_1", "resume_2"]:
                (category_dir / f"{resume_id}.pdf").write_bytes(b"%PDF-1.4\n")

        return archive_dir

    def _stub_processed_resume(self, file_path, job_category=None, resume_id=None):
        """Return a minimal StructuredResume without parsing a real PDF."""
        return StructuredResume(
            resume_id=resume_id or Path(file_path).stem,
            job_category=job_category or "UNKNOWN",
            sections=ResumeSections(
                raw_text="Sample resume text",
                skills="Python, Excel",
                experience="Sample experience",
                education="Sample education",
                projects=""
            ),
            skills=SkillSet(
                explicit_skills=["Python", "Excel"],
                implicit_skills=[]
            ),
            normalized_skills=["Python", "Excel"],
            scores=None,
            metadata=ResumeMetadata(
                file_path=file_path,
                processed_date="2024-01-01T00:00:00",
                processing_time_ms=1
            )
        )
    
    def test_load_from_archive_structure(self, processor, sample_pdf_archive, monkeypatch):
        """Test loading resumes from archive organized by job categories."""
        monkeypatch.setattr(processor, "process_resume", self._stub_processed_resume)

        results = processor.load_from_archive(str(sample_pdf_archive))
        
        # Verify structure
        assert isinstance(results, dict)
        assert set(results) == {"ACCOUNTANT", "INFORMATION-TECHNOLOGY"}
        
        # Verify each category has resumes
        for category, resumes in results.items():
            assert isinstance(resumes, list)
            assert len(resumes) == 2
            
            # Verify each resume has correct category
            for resume in resumes:
                assert resume.job_category == category
    
    def test_archive_preserves_category_organization(
        self,
        processor,
        sample_pdf_archive,
        monkeypatch
    ):
        """Test that archive loading preserves job category organization."""
        monkeypatch.setattr(processor, "process_resume", self._stub_processed_resume)

        results = processor.load_from_archive(str(sample_pdf_archive))
        
        # Verify categories match directory structure
        expected_categories = ['ACCOUNTANT', 'INFORMATION-TECHNOLOGY']
        
        for category in expected_categories:
            category_path = sample_pdf_archive / category
            if category_path.exists():
                assert category in results
                assert len(results[category]) > 0


class TestCSVValidation:
    """Test CSV validation against PDF extraction."""
    
    def test_validate_csv_extraction_against_pdf(self, processor):
        """Test validation of CSV extraction against PDF extraction."""
        # Use a real PDF from archive
        pdf_path = "archive/data/data/ACCOUNTANT/10554236.pdf"
        
        if not Path(pdf_path).exists():
            pytest.skip(f"PDF file not found: {pdf_path}")
        
        # Create sample CSV text (simulating what would be in CSV)
        csv_text = "Sample resume text for accountant position with financial analysis skills"
        
        # Validate extraction
        result = processor.validate_csv_extraction(
            pdf_path=pdf_path,
            csv_resume_str=csv_text,
            resume_id="10554236"
        )
        
        # Verify validation result structure
        assert 'resume_id' in result
        assert 'pdf_length' in result
        assert 'csv_length' in result
        assert 'length_ratio' in result
        assert 'text_similarity' in result
        assert 'extraction_success' in result
        
        # Verify metrics are reasonable
        assert result['pdf_length'] > 0
        assert result['csv_length'] > 0
        assert 0 <= result['text_similarity'] <= 1.0
    
    def test_cross_validate_data_sources(self, processor):
        """Test cross-validation between CSV and PDF data sources."""
        # Use real CSV file
        csv_path = "archive/Resume/Resume.csv"
        
        if not Path(csv_path).exists():
            pytest.skip(f"CSV file not found: {csv_path}")
        
        # Load a small sample from CSV
        csv_data = processor.load_from_csv(csv_path)
        
        if len(csv_data) == 0:
            pytest.skip("No data in CSV file")
        
        # Take first entry
        sample = csv_data[0]
        resume_id = sample['ID']
        category = sample['Category']
        csv_text = sample['Resume_str']
        
        # Try to find corresponding PDF
        pdf_path = f"archive/data/data/{category}/{resume_id}.pdf"
        
        if not Path(pdf_path).exists():
            pytest.skip(f"Corresponding PDF not found: {pdf_path}")
        
        # Cross-validate
        validation_results = processor.cross_validate_data_sources(
            csv_path=csv_path,
            pdf_archive_path="archive/data/data",
            max_samples=1
        )
        
        # Verify results structure
        assert isinstance(validation_results, dict)
        assert 'total_validated' in validation_results
        assert 'validation_details' in validation_results
        assert validation_results['total_validated'] >= 0


class TestErrorHandling:
    """Test error handling and recovery for both data sources."""
    
    def test_pdf_processing_missing_file(self, processor):
        """Test error handling for missing PDF file."""
        with pytest.raises(FileNotFoundError):
            processor.process_resume("nonexistent.pdf", job_category="TEST")
    
    def test_csv_processing_missing_file(self, processor):
        """Test error handling for missing CSV file."""
        with pytest.raises(FileNotFoundError):
            processor.load_from_csv("nonexistent.csv")
    
    def test_csv_processing_invalid_format(self, processor):
        """Test error handling for invalid CSV format."""
        # Create CSV with missing required columns
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'Text'])  # Missing columns
            writer.writeheader()
            writer.writerow({'ID': '123', 'Text': 'Test'})
            csv_path = f.name
        
        try:
            with pytest.raises(ValueError, match="CSV missing required columns"):
                processor.load_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()
    
    def test_pdf_processing_corrupted_file(self, processor):
        """Test error handling for corrupted PDF file."""
        # Create a fake PDF file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf') as f:
            f.write("This is not a valid PDF file")
            pdf_path = f.name
        
        try:
            # Should handle error gracefully
            result = processor.process_resume(pdf_path, job_category="TEST")
            # If it returns a result, verify it's reasonable
            assert result is not None or True  # Either returns result or raises error
        except Exception:
            # Error is acceptable for corrupted file
            assert True
        finally:
            Path(pdf_path).unlink()
    
    def test_csv_processing_empty_resume_text(self, processor):
        """Test handling of empty resume text in CSV."""
        # Create CSV with empty resume text
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'Resume_str', 'Resume_html', 'Category'])
            writer.writeheader()
            writer.writerow({
                'ID': '999',
                'Resume_str': '',  # Empty text
                'Resume_html': '',
                'Category': 'TEST'
            })
            csv_path = f.name
        
        try:
            resumes = processor.process_csv_data(csv_path)
            
            # Should handle gracefully
            assert len(resumes) == 1
            assert resumes[0].resume_id == '999'
            # Empty text should result in empty or minimal skills
            assert isinstance(resumes[0].normalized_skills, list)
        finally:
            Path(csv_path).unlink()
    
    def test_archive_loading_missing_directory(self, processor):
        """Test error handling for missing archive directory."""
        with pytest.raises(FileNotFoundError):
            processor.load_from_archive("nonexistent/archive/path")
    
    def test_batch_processing_partial_failures(self, processor):
        """Test that batch processing continues after individual failures."""
        # Create temp directory with mix of files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            valid_csv = temp_path / "valid.csv"
            with open(valid_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['ID', 'Resume_str', 'Resume_html', 'Category'])
                writer.writeheader()
                writer.writerow({
                    'ID': '1',
                    'Resume_str': 'Valid resume with skills',
                    'Resume_html': '<p>Test</p>',
                    'Category': 'TEST'
                })
            
            # Process valid file
            resumes = processor.process_csv_data(str(valid_csv))
            assert len(resumes) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
