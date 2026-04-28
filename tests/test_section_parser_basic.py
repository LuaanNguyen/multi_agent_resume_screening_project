"""
Basic verification tests for SectionParser implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.section_parser import SectionParser
from src.models import ResumeSections


def test_section_parser_initialization():
    """Test that SectionParser initializes correctly with default patterns."""
    parser = SectionParser()
    assert parser is not None
    assert 'skills' in parser.section_patterns
    assert 'experience' in parser.section_patterns
    assert 'education' in parser.section_patterns
    assert 'projects' in parser.section_patterns


def test_parse_sections_with_all_sections():
    """Test parsing resume with all sections present."""
    parser = SectionParser()
    
    resume_text = """
    John Doe
    Software Engineer
    
    SKILLS
    Python, JavaScript, React, Machine Learning
    
    EXPERIENCE
    Senior Developer at Tech Corp
    - Built scalable applications
    - Led team of 5 developers
    
    EDUCATION
    BS Computer Science, MIT, 2015
    
    PROJECTS
    E-commerce Platform - Built using React and Node.js
    """
    
    sections = parser.parse_sections(resume_text)
    
    assert isinstance(sections, ResumeSections)
    assert "Python" in sections.skills
    assert "JavaScript" in sections.skills
    assert "Senior Developer" in sections.experience
    assert "Tech Corp" in sections.experience
    assert "Computer Science" in sections.education
    assert "MIT" in sections.education
    assert "E-commerce Platform" in sections.projects
    assert sections.raw_text == resume_text


def test_parse_sections_with_missing_sections():
    """Test parsing resume with some sections missing."""
    parser = SectionParser()
    
    resume_text = """
    Jane Smith
    Data Scientist
    
    Skills
    Python, R, SQL, TensorFlow
    
    Experience
    Data Analyst at Analytics Inc
    - Analyzed customer data
    """
    
    sections = parser.parse_sections(resume_text)
    
    assert isinstance(sections, ResumeSections)
    assert "Python" in sections.skills
    assert "Data Analyst" in sections.experience
    assert sections.education == ""  # Missing section
    assert sections.projects == ""   # Missing section


def test_extract_section_case_insensitive():
    """Test that section extraction is case-insensitive."""
    parser = SectionParser()
    
    resume_text = """
    SKILLS
    Java, C++
    
    experience
    Software Engineer at StartupXYZ
    
    EDUCATION
    MS in CS
    """
    
    skills = parser.extract_section(resume_text, 'skills')
    experience = parser.extract_section(resume_text, 'experience')
    
    assert "Java" in skills
    assert "C++" in skills
    assert "Software Engineer" in experience
    assert "StartupXYZ" in experience


def test_extract_section_with_variations():
    """Test section extraction with header variations."""
    parser = SectionParser()
    
    # Test "Technical Skills" variation
    resume_text1 = """
    Technical Skills
    Python, Docker, Kubernetes
    
    Work History
    DevOps Engineer
    """
    
    skills = parser.extract_section(resume_text1, 'skills')
    assert "Python" in skills
    assert "Docker" in skills
    
    # Test "Work History" variation for experience
    experience = parser.extract_section(resume_text1, 'experience')
    assert "DevOps Engineer" in experience


def test_extract_section_not_found():
    """Test that empty string is returned when section is not found."""
    parser = SectionParser()
    
    resume_text = """
    John Doe
    Some content without proper sections
    """
    
    skills = parser.extract_section(resume_text, 'skills')
    experience = parser.extract_section(resume_text, 'experience')
    
    assert skills == ""
    assert experience == ""


def test_section_boundaries():
    """Test that sections are properly bounded and don't overlap."""
    parser = SectionParser()
    
    resume_text = """
    SKILLS
    Python, Java, C++
    
    EXPERIENCE
    Software Engineer at TechCo
    Developed applications
    
    EDUCATION
    BS Computer Science
    """
    
    skills = parser.extract_section(resume_text, 'skills')
    experience = parser.extract_section(resume_text, 'experience')
    education = parser.extract_section(resume_text, 'education')
    
    # Skills should not contain experience content
    assert "Software Engineer" not in skills
    assert "TechCo" not in skills
    
    # Experience should not contain education content
    assert "BS Computer Science" not in experience
    
    # Each section should contain its own content
    assert "Python" in skills
    assert "Software Engineer" in experience
    assert "Computer Science" in education


if __name__ == "__main__":
    # Run basic tests
    test_section_parser_initialization()
    print("[PASS] Initialization test passed")
    
    test_parse_sections_with_all_sections()
    print("[PASS] Parse all sections test passed")
    
    test_parse_sections_with_missing_sections()
    print("[PASS] Parse missing sections test passed")
    
    test_extract_section_case_insensitive()
    print("[PASS] Case insensitive test passed")
    
    test_extract_section_with_variations()
    print("[PASS] Header variations test passed")
    
    test_extract_section_not_found()
    print("[PASS] Section not found test passed")
    
    test_section_boundaries()
    print("[PASS] Section boundaries test passed")
    
    print("\n[PASS] All basic tests passed!")
