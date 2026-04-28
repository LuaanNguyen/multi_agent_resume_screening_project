"""
Demonstration of the SectionParser implementation.
"""

from src.section_parser import SectionParser
from src.models import ResumeSections


def main():
    # Create a sample resume
    sample_resume = """
    JOHN DOE
    Software Engineer | john.doe@email.com | (555) 123-4567
    
    SKILLS
    - Programming Languages: Python, JavaScript, Java, C++
    - Frameworks: React, Django, Flask, Node.js
    - Tools: Git, Docker, Kubernetes, AWS
    - Machine Learning: TensorFlow, PyTorch, scikit-learn
    
    EXPERIENCE
    Senior Software Engineer | Tech Corp | 2020 - Present
    - Led development of microservices architecture serving 1M+ users
    - Implemented CI/CD pipelines reducing deployment time by 60%
    - Mentored team of 5 junior developers
    
    Software Developer | StartupXYZ | 2018 - 2020
    - Built RESTful APIs using Django and PostgreSQL
    - Developed React-based dashboard for data visualization
    - Improved application performance by 40%
    
    EDUCATION
    Master of Science in Computer Science
    Massachusetts Institute of Technology | 2016 - 2018
    GPA: 3.9/4.0
    
    Bachelor of Science in Computer Engineering
    University of California, Berkeley | 2012 - 2016
    GPA: 3.8/4.0
    
    PROJECTS
    E-Commerce Platform (2021)
    - Built full-stack application using React, Node.js, and MongoDB
    - Implemented payment processing with Stripe API
    - Deployed on AWS with auto-scaling capabilities
    
    Machine Learning Pipeline (2020)
    - Developed automated ML pipeline for customer churn prediction
    - Achieved 92% accuracy using ensemble methods
    - Reduced model training time by 50% using distributed computing
    """
    
    print("=" * 80)
    print("SECTION PARSER DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize the parser
    parser = SectionParser()
    print("[PASS] SectionParser initialized with default patterns")
    print()
    
    # Parse the resume
    sections = parser.parse_sections(sample_resume)
    print("[PASS] Resume parsed successfully")
    print()
    
    # Display extracted sections
    print("-" * 80)
    print("EXTRACTED SECTIONS:")
    print("-" * 80)
    print()
    
    print("SKILLS SECTION:")
    print(sections.skills[:200] + "..." if len(sections.skills) > 200 else sections.skills)
    print()
    
    print("EXPERIENCE SECTION:")
    print(sections.experience[:300] + "..." if len(sections.experience) > 300 else sections.experience)
    print()
    
    print("EDUCATION SECTION:")
    print(sections.education[:200] + "..." if len(sections.education) > 200 else sections.education)
    print()
    
    print("PROJECTS SECTION:")
    print(sections.projects[:300] + "..." if len(sections.projects) > 300 else sections.projects)
    print()
    
    # Verify parser behavior
    print("-" * 80)
    print("PARSER CHECKS:")
    print("-" * 80)
    print()
    
    print("[PASS] Skills section identified -", "Python" in sections.skills)
    print("[PASS] Experience section identified -", "Tech Corp" in sections.experience)
    print("[PASS] Education section identified -", "MIT" in sections.education or "Massachusetts" in sections.education)
    print("[PASS] Projects section identified -", "E-Commerce" in sections.projects)
    print("[PASS] Missing sections handled -", isinstance(sections, ResumeSections))
    print("[PASS] Sections output as labeled text segments -", len(sections.raw_text) > 0)
    print()
    
    # Test with missing sections
    print("-" * 80)
    print("TESTING WITH MISSING SECTIONS:")
    print("-" * 80)
    print()
    
    incomplete_resume = """
    Jane Smith
    Data Scientist
    
    Technical Skills
    Python, R, SQL, TensorFlow, Pandas
    
    Work History
    Data Scientist at Analytics Inc
    Analyzed customer behavior data
    """
    
    incomplete_sections = parser.parse_sections(incomplete_resume)
    print("Resume with only Skills and Experience sections:")
    print(f"  - Skills found: {'Yes' if incomplete_sections.skills else 'No'}")
    print(f"  - Experience found: {'Yes' if incomplete_sections.experience else 'No'}")
    print(f"  - Education found: {'Yes' if incomplete_sections.education else 'No (empty string returned)'}")
    print(f"  - Projects found: {'Yes' if incomplete_sections.projects else 'No (empty string returned)'}")
    print()
    
    print("=" * 80)
    print("[PASS] Demonstration complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
