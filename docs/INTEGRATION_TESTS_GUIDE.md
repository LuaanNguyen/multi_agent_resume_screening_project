# Integration Tests Guide

This guide documents the integration tests for the command-line workflow in `main.py`.

## Test File

- File: `tests/test_main_integration.py`
- Coverage focus: CLI argument parsing, CSV/PDF processing, model training, evaluation, association mining, clustering, validation reports, and output-file generation.

## Quick Start

Run the integration test script:

```bash
./run_integration_tests.sh
```

Or run the tests directly:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest tests/test_main_integration.py -v
```

Run selected tests:

```bash
pytest tests/test_main_integration.py::TestCSVProcessingPipeline -v
pytest tests/test_main_integration.py::TestCSVProcessingPipeline::test_csv_to_clustering_pipeline -v
pytest tests/test_main_integration.py::TestCrossSourceValidation -v
pytest tests/test_main_integration.py::TestOutputFileGeneration -v
```

## What The Tests Cover

The integration suite exercises these command paths:

1. `process-csv`
2. `process-pdf`
3. `train`
4. `evaluate`
5. `mine`
6. `cluster`
7. `validate`

The tests verify that these workflows can generate the expected local artifacts:

1. structured resume JSON files
2. trained model files
3. vocabulary files
4. evaluation reports
5. association-rule reports
6. cluster reports and assignments
7. validation reports

## Fixture Data

The suite uses small local fixtures rather than the full Kaggle dataset:

- temporary CSV files created during tests
- sample text resumes under `tests/fixtures/sample_resumes/`
- expected structured outputs under `tests/fixtures/expected_outputs/`
- temporary output directories that are cleaned after each test

These fixtures are intended for smoke and integration coverage. They are not intended to produce meaningful model-quality metrics.

## Recommended Commands

Run the main integration tests:

```bash
pytest tests/test_main_integration.py -q
```

Run the evaluation and association modules used by the CLI:

```bash
pytest tests/test_evaluation_module.py tests/test_association_miner.py -q
```

Run the full suite when dependencies are installed:

```bash
pytest -q
```

## Notes

- Tests that use the NLP pipeline require `en_core_web_sm`.
- The real Kaggle dataset is downloaded with `python setup_dataset.py` and is intentionally excluded from Git.
- Generated outputs under `output/` are intentionally excluded from Git.
