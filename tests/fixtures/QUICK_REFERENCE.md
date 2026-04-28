# Test Fixtures Quick Reference

## Directory Structure

```
tests/fixtures/
├── sample_resumes/pdf/          # 3 sample resume files
├── sample_data/                 # CSV with 5 entries
├── expected_outputs/            # 2 expected JSON outputs
├── job_requirements/            # 3 job requirement files
└── validation_data/             # PDF-CSV mapping
```

## Available Fixtures

| Fixture Name | Returns | Usage |
|--------------|---------|-------|
| `fixtures_dir` | Path to fixtures directory | `fixtures_dir / "my_file.json"` |
| `sample_csv_path` | Path to sample CSV | `pd.read_csv(sample_csv_path)` |
| `sample_pdf_resumes` | Dict of PDF paths | `sample_pdf_resumes["accountant"]` |
| `expected_outputs` | Dict of expected output paths | `expected_outputs["accountant"]` |
| `job_requirements` | Dict of job requirement paths | `job_requirements["accountant"]` |
| `validation_mapping` | Path to validation mapping | `open(validation_mapping)` |
| `load_job_requirements` | Function to load job reqs | `load_job_requirements("accountant")` |
| `load_expected_output` | Function to load expected | `load_expected_output("accountant")` |

## Sample Data Keys

### PDF Resumes
- `"accountant"` - CPA with 8+ years
- `"it_developer"` - Full Stack Developer
- `"healthcare"` - Registered Nurse

### Expected Outputs
- `"accountant"` - Accountant expected JSON
- `"it_developer"` - IT Developer expected JSON

### Job Requirements
- `"accountant"` - Accountant job
- `"software_developer"` - Developer job
- `"nurse"` - Nurse job

## Quick Examples

### Load CSV Data
```python
def test_csv(sample_csv_path):
    df = pd.read_csv(sample_csv_path)
    # 5 entries: TEST001-TEST005
```

### Load PDF Resume
```python
def test_pdf(sample_pdf_resumes):
    text = sample_pdf_resumes["accountant"].read_text()
```

### Load Job Requirements
```python
def test_job_req(load_job_requirements):
    req = load_job_requirements("accountant")
    skills = req["required_skills"]
```

### Load Expected Output
```python
def test_expected(load_expected_output):
    expected = load_expected_output("accountant")
    assert expected["job_category"] == "ACCOUNTANT"
```

### Cross-Source Validation
```python
def test_validation(validation_mapping):
    with open(validation_mapping) as f:
        data = json.load(f)
    mappings = data["mappings"]  # 3 mappings
```

## CSV Data Structure

| ID | Category | Content |
|----|----------|---------|
| TEST001 | ACCOUNTANT | Jane Doe, Financial skills |
| TEST002 | INFORMATION-TECHNOLOGY | Mike Chen, Tech skills |
| TEST003 | HEALTHCARE | Emily Brown, Nursing skills |
| TEST004 | ENGINEERING | Robert Lee, Mechanical skills |
| TEST005 | SALES | Lisa Anderson, Sales skills |

## Validation

Run fixture tests:
```bash
python3 -m pytest tests/test_fixtures.py -v
```

Expected: 33 tests pass

## Full Documentation

- **README.md** - Overview and structure
- **USAGE_GUIDE.md** - Detailed examples and patterns
- **QUICK_REFERENCE.md** - This file

## Tips

1. Use fixtures instead of hardcoded paths
2. Leverage helper functions (`load_*`)
3. Test both CSV and PDF pipelines
4. Validate cross-source consistency
5. Check `test_fixtures.py` for examples

## Related Files

- `tests/conftest.py` - Fixture definitions
- `tests/test_fixtures.py` - Fixture validation tests
- `config/skill_aliases.json` - Skill normalization
- `config/job_categories.json` - Valid categories
