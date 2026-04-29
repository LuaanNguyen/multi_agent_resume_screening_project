# Real Data Results

This file summarizes the latest successful run on the Kaggle Resume Dataset using the code currently in this repository.

## Dataset Summary

- CSV resumes: `2484`
- PDF files in archive: `2484`
- Job categories: `24`
- Successfully compared CSV/PDF resumes in validation: `2483`

Commands used:

```bash
python setup_dataset.py
python main.py --output-dir output process-csv --csv-file archive/Resume/Resume.csv
python main.py --output-dir output train --csv-file archive/Resume/Resume.csv
python main.py --output-dir output evaluate --csv-file archive/Resume/Resume.csv
python main.py --output-dir output mine --csv-file archive/Resume/Resume.csv
python main.py --output-dir output cluster --source csv --csv-file archive/Resume/Resume.csv
python main.py --output-dir output cluster --source pdf --pdf-dir archive/data/data
python main.py --output-dir output validate --csv-file archive/Resume/Resume.csv --pdf-dir archive/data/data
```

Generated artifacts:

- `output/models/classifier.pkl`
- `output/models/feature_generator.pkl`
- `output/models/vocabulary.json`
- `output/reports/evaluation_report.json`
- `output/reports/association_rules.json`
- `output/reports/cluster_report.json`
- `output/reports/cluster_assignments.json`
- `output/reports/validation_report.json`

## Classification Results

The current real-data run uses the configuration in `config/config.yaml`, including:

- `test_size = 0.2`
- `random_state = 42`

Observed metrics:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Baseline: TF-IDF + Logistic Regression | 0.6258 | 0.5468 |
| Proposed: Hybrid Text + Skill Logistic Regression | 0.4165 | 0.3685 |

Observation:

- The baseline model outperformed the proposed hybrid text-plus-skill model in the current implementation.

Fairness output for the proposed model:

- Mean per-category F1: `0.3685`
- F1 standard deviation: `0.1870`
- Flagged categories: `AGRICULTURE`, `APPAREL`, `ARTS`, `BPO`, `CONSULTANT`

Best proposed-model categories by F1:

- `CHEF`: `0.7391`
- `CONSTRUCTION`: `0.6441`
- `ACCOUNTANT`: `0.6182`
- `TEACHER`: `0.5455`
- `INFORMATION-TECHNOLOGY`: `0.5246`

## Association Mining Results

Rule count: `0`

Mining details:

- Non-empty cleaned transactions: `2482`
- Frequent itemsets found before confidence filtering: `21`
- Configured `min_support`: `0.1`
- Configured `min_confidence`: `0.5`

Observation:

- Default transaction cleanup removes obvious resume header/contact/location artifacts before Apriori. With the current confidence threshold, no rules met the threshold on the full dataset.

## Clustering Results

CSV source:

- Samples clustered: `2484`
- Requested clusters: `10`
- Actual clusters: `10`
- Silhouette score: `0.1024`
- Cluster sizes: `158`, `1`, `461`, `372`, `1`, `1`, `2`, `1`, `1`, `1486`

PDF source:

- Samples clustered: `2483`
- Requested clusters: `10`
- Actual clusters: `10`
- Silhouette score: `0.1620`
- Cluster sizes: `1`, `442`, `1`, `246`, `1`, `1`, `1`, `1788`, `1`, `1`

Note:

- The cluster command writes `output/reports/cluster_report.json` and `output/reports/cluster_assignments.json`. Running the PDF clustering command after CSV clustering overwrites those paths with the PDF result.

Observation:

- The CSV and PDF clustering results are exploratory. Each run produced one dominant cluster and several singleton or near-singleton clusters, so the current feature space does not produce balanced groups.

## Cross-Source Validation Results

Observed metrics:

- Samples compared: `2483`
- Text similarity mean: `0.9979`
- Text similarity std: `0.0172`
- Skill overlap mean: `0.7575`
- Skill overlap std: `0.1633`
- Extraction accuracy: `0.8777`

Observation:

- PDF text extraction is very close to the CSV source text.
- Skill extraction is less stable than raw text extraction, which is reflected in the lower skill-overlap score.

## Current Reporting Guidance

If you reference this project in a course report or README, the safest claims are:

- The CLI workflow `process-csv`, `process-pdf`, `train`, `evaluate`, `mine`, `cluster`, and `validate` runs successfully on the Kaggle dataset.
- The `cluster` CLI runs on CSV and PDF sources and writes aggregate plus per-resume cluster reports.
- The repository currently supports 24 job categories from the Kaggle dataset.
- The baseline classifier outperformed the proposed hybrid classifier in the latest real-data run.
- Cross-source extraction validation produced strong text similarity and moderate skill overlap.
- The scoring engine exists in code, but normal CLI processing still saves `scores: null` in structured resumes.
