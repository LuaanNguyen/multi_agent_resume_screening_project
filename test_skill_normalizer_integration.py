"""
Integration test for SkillNormalizer with actual config file.

This demonstrates the SkillNormalizer working with the real skill_aliases.json file.
"""

from src.skill_normalizer import SkillNormalizer


def test_skill_normalizer_integration():
    """Test SkillNormalizer with actual config file."""
    
    # Create normalizer and load from config
    normalizer = SkillNormalizer({})
    alias_dict = normalizer.load_alias_dictionary("config/skill_aliases.json")
    
    # Create new normalizer with loaded dictionary
    normalizer = SkillNormalizer(alias_dict, fuzzy_threshold=85)
    
    # Test various skill normalizations
    test_cases = [
        # Exact matches
        ("js", "JavaScript"),
        ("react.js", "React"),
        ("ml", "Machine Learning"),
        ("py", "Python"),
        ("k8s", "Kubernetes"),
        
        # Case variations
        ("JS", "JavaScript"),
        ("PYTHON3", "Python"),
        ("Sql", "SQL"),
        
        # Fuzzy matches (typos)
        ("Pyton", "Python"),
        ("JavaScritp", "JavaScript"),  # transposed letters
        ("Reactjs", "React"),
        
        # Unknown skills (fallback)
        ("UnknownSkill", "unknownskill"),
        ("NewTechnology", "newtechnology"),
    ]
    
    print("\n=== SkillNormalizer Integration Test ===\n")
    
    for input_skill, expected_output in test_cases:
        result = normalizer.normalize_skill(input_skill)
        status = "[PASS]" if result == expected_output else "[FAIL]"
        print(f"{status} '{input_skill}' -> '{result}' (expected: '{expected_output}')")
        assert result == expected_output, f"Failed: {input_skill} -> {result} (expected {expected_output})"
    
    # Test list normalization
    skills_list = ["js", "Python3", "react.js", "Machine-Learning", "UnknownSkill"]
    normalized = normalizer.normalize_skills(skills_list)
    
    print(f"\n=== List Normalization ===")
    print(f"Input:  {skills_list}")
    print(f"Output: {normalized}")
    
    expected_normalized = ["JavaScript", "Python", "React", "Machine Learning", "unknownskill"]
    assert normalized == expected_normalized
    
    print("\n[PASS] All integration tests passed!")


if __name__ == "__main__":
    test_skill_normalizer_integration()
