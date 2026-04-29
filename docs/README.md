# Documentation Index

This folder contains the supporting documentation for the resume screening and
skill mining project. The root `README.md` is the main submission landing page;
the files here provide deeper implementation, command, and result details.

## Recommended Reading Order

| File | Use It For |
| --- | --- |
| [`REAL_DATA_RESULTS.md`](REAL_DATA_RESULTS.md) | Real Kaggle dataset metrics and reporting guidance |
| [`CLI_REFERENCE.md`](CLI_REFERENCE.md) | Complete command-line usage and output reference |
| [`RESUME_PROCESSOR_ARCHITECTURE.md`](RESUME_PROCESSOR_ARCHITECTURE.md) | Pipeline architecture and processing flow |
| [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) | Developer-facing module and class details |
| [`INTEGRATION_TESTS_GUIDE.md`](INTEGRATION_TESTS_GUIDE.md) | Integration test scope and commands |

## Local Web Dashboard

The project includes an optional FastAPI dashboard over the same CLI pipeline.
Start it from the repository root:

```bash
python -m uvicorn webapp.app:app --reload
```

Then open `http://127.0.0.1:8000`. The dashboard displays generated report
files and can launch the existing local pipeline commands.

## Local Generated Files

Generated files are intentionally kept outside Git:

- `archive/` contains the downloaded Kaggle dataset.
- `output/` contains processed resumes, models, and reports.
- `.venv/` contains the local Python environment.

Use the root `README.md` setup and workflow commands to recreate these local
artifacts when needed.
