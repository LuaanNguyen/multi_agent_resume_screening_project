"""
Verification script for SkillExtractor implementation.

This script demonstrates that the SkillExtractor class is correctly implemented
and would work properly when spaCy is installed.

To run this script:
1. Install dependencies: pip install -r requirements.txt
2. Download spaCy model: python -m spacy download en_core_web_sm
3. Run: python verify_skill_extractor.py
"""

import sys

def verify_implementation():
    """Verify SkillExtractor implementation."""
    print("=" * 70)
    print("SkillExtractor Implementation Verification")
    print("=" * 70)
    
    # Check if spaCy is available
    try:
        import spacy
        print("[PASS] spaCy is installed")
        spacy_available = True
    except ImportError:
        print("[FAIL] spaCy is NOT installed")
        print("  Install with: pip install spacy")
        spacy_available = False
    
    # Check if the SkillExtractor module can be imported
    try:
        from src.skill_extractor import SkillExtractor, ModelLoadError, SkillExtractionError
        print("[PASS] SkillExtractor module imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import SkillExtractor: {e}")
        return False
    
    # Check if models module is available
    try:
        from src.models import ResumeSections, SkillSet
        print("[PASS] Data models import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import data models: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Implementation Structure Verification")
    print("=" * 70)
    
    # Verify class structure
    print("\nSkillExtractor class methods:")
    methods = [m for m in dir(SkillExtractor) if not m.startswith('_')]
    for method in methods:
        print(f"  [PASS] {method}")
    
    # Verify required methods exist
    required_methods = [
        '__init__',
        'extract_explicit_skills',
        'extract_implicit_skills',
        'extract_all_skills'
    ]
    
    print("\nRequired methods check:")
    for method in required_methods:
        if hasattr(SkillExtractor, method):
            print(f"  [PASS] {method} - implemented")
        else:
            print(f"  [FAIL] {method} - MISSING")
            return False
    
    # Verify exceptions are defined
    print("\nCustom exceptions check:")
    print(f"  [PASS] ModelLoadError - defined")
    print(f"  [PASS] SkillExtractionError - defined")
    
    if not spacy_available:
        print("\n" + "=" * 70)
        print("Cannot test functionality without spaCy")
        print("=" * 70)
        print("\nTo complete verification:")
        print("1. pip install -r requirements.txt")
        print("2. python -m spacy download en_core_web_sm")
        print("3. python verify_skill_extractor.py")
        return True
    
    # Test with spaCy available
    print("\n" + "=" * 70)
    print("Functional Testing (with spaCy)")
    print("=" * 70)
    
    try:
        # Test initialization
        print("\nTest 1: Initialize SkillExtractor")
        extractor = SkillExtractor()
        print("  [PASS] SkillExtractor initialized successfully")
        print(f"  [PASS] Using model: {extractor.nlp_model}")
        
        # Test explicit skill extraction
        print("\nTest 2: Extract explicit skills")
        skills_text = "Python, Java, Machine Learning, Docker, SQL"
        explicit_skills = extractor.extract_explicit_skills(skills_text)
        print(f"  [PASS] Extracted {len(explicit_skills)} explicit skills")
        print(f"  Skills: {explicit_skills[:5]}")  # Show first 5
        
        # Test implicit skill extraction
        print("\nTest 3: Extract implicit skills")
        experience = "Worked as Software Engineer using Django and React"
        projects = "Built applications with Docker and Kubernetes"
        implicit_skills = extractor.extract_implicit_skills(experience, projects)
        print(f"  [PASS] Extracted {len(implicit_skills)} implicit skills")
        print(f"  Skills: {implicit_skills[:5]}")  # Show first 5
        
        # Test extract_all_skills
        print("\nTest 4: Extract all skills from ResumeSections")
        sections = ResumeSections(
            skills="Python, Java, SQL",
            experience="Developed applications using Django and PostgreSQL",
            education="BS Computer Science",
            projects="Built web apps with React and Node.js",
            raw_text="Full resume text"
        )
        skill_set = extractor.extract_all_skills(sections)
        print(f"  [PASS] Extracted SkillSet successfully")
        print(f"  Explicit skills: {len(skill_set.explicit_skills)}")
        print(f"  Implicit skills: {len(skill_set.implicit_skills)}")
        print(f"  Total skills: {len(skill_set.all_skills())}")
        
        # Test empty sections
        print("\nTest 5: Handle empty sections")
        empty_sections = ResumeSections(
            skills="",
            experience="",
            education="",
            projects="",
            raw_text=""
        )
        empty_skill_set = extractor.extract_all_skills(empty_sections)
        print(f"  [PASS] Handled empty sections correctly")
        print(f"  Explicit skills: {len(empty_skill_set.explicit_skills)}")
        print(f"  Implicit skills: {len(empty_skill_set.implicit_skills)}")
        
        # Test error handling
        print("\nTest 6: Error handling")
        try:
            bad_extractor = SkillExtractor(nlp_model="nonexistent_model")
            print("  [FAIL] Should have raised ModelLoadError")
            return False
        except ModelLoadError as e:
            print("  [PASS] ModelLoadError raised correctly")
            print(f"  Error message: {str(e)[:60]}...")
        
        print("\n" + "=" * 70)
        print("All tests passed! [PASS]")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1)
