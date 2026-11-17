# LLM Regression Detection System

A Python-based evaluation framework for measuring how changes in prompts, datasets, and model configuration affect LLM behavior over time.

This project is designed to answer a practical engineering question: when a model or prompt changes, does the system improve, regress, or remain statistically equivalent on real evaluation cases? The framework evaluates a versioned golden dataset, compares checkpoint results across prompt versions, and surfaces meaningful regressions before they reach production.

**[Evaluation Dashboard](https://manvitha-kv352.github.io/Model-Regression-Detection-System/)** | **[Source Code](https://github.com/Manvitha-kv352/Model-Regression-Detection-System)**

---

## Overview

LLM quality is rarely captured well by a single aggregate metric. A new prompt can increase overall accuracy while still introducing regressions on previously correct edge cases. This project addresses that gap by combining case-level evaluation, checkpoint comparison, statistical testing, and operational guardrails.

The evaluation workload uses a customer-support email classification task. Each email is labeled into one of four categories: billing, technical, account, or general. In addition to the final class, the model generates a brief summary of the customer request, which is also checked for correctness and consistency.

The result is a repeatable pipeline for:

- evaluating prompt and dataset changes against a fixed golden set
- identifying individual test-case regressions and improvements
- comparing baseline and candidate runs with statistical context
- enforcing quality gates before accepting a model change
- monitoring long-term drift and degradation in production-like conditions

---

## System Architecture

```text
        Prompt Version / Configuration
                    |
                    v
              Golden Dataset
                    |
                    v
            Evaluation Runner (async)
                    |
                    v
                 Groq API
                    |
                    v
             Checkpointed Results
                    |
                    v
        Regression Comparator
            /                   \
           /                     \
          v                       v
   Statistical Analysis       Quality Gate
          |                       |
          +-----------+-----------+
                      |
                      v
              Evaluation Reports
                      |
                      v
        Monitoring / Alerts / Slack
```

The system is built around a small but reliable workflow: generate versioned prompts and datasets, execute model evaluations, compare results across checkpoints, and publish a reviewable artifacts package for engineering decisions.

---

## Core Components

### Dataset and Prompt Versioning

The project stores evaluation inputs in versioned JSON datasets and prompt templates in separate YAML configuration files. This makes it possible to compare behavior across iterations without changing the underlying test set or losing traceability of the prompt used for each run.

The current benchmark contains 30 structured test cases, including straightforward requests and examples where more than one category could appear relevant. This is important because a model may appear strong on average while still failing on ambiguous or nuanced support tickets.

### Asynchronous Evaluation Runner

The evaluation runner executes inference in batches using Python's `asyncio` and the Groq API. It records classification results, generated summaries, latency, token usage, and execution errors for each case.

Key reliability features include:

- checkpoint persistence after each batch
- resumable runs for interrupted evaluations
- SHA-256 fingerprinting of prompt and dataset content to prevent accidental reuse of mismatched results
- structured outputs suitable for reporting and downstream comparison

### Regression Comparison

The regression comparator analyzes baseline and candidate checkpoints to identify case-level changes in behavior. It distinguishes between:

- improved examples
- regressed examples
- unchanged examples
- category-level accuracy deltas

This is particularly useful because aggregate accuracy alone can hide important qualitative changes. Two prompt variants may score identically overall while producing different error patterns.

### Statistical Testing

The project integrates the exact McNemar test to assess whether observed classification differences between baseline and candidate evaluations are statistically meaningful. This adds an analytical layer beyond raw accuracy and helps interpret whether a change is likely systematic or incidental.

### Quality Gates

The quality gate module evaluates candidate results against configurable regression thresholds and generates a structured decision report. It considers both overall accuracy changes and individual test-case regressions.

The report helps determine whether a prompt change requires further review. Quality-gate decisions are not yet enforced as mandatory merge-blocking checks in GitHub Actions.

### Monitoring and Alerting

I implemented rolling-window accuracy monitoring to track model performance across evaluation runs. The monitoring module identifies stable performance, potential degradation, or insufficient evaluation history based on configurable thresholds.

I tested the drift-detection logic using simulated evaluation histories to verify how the system responds to different accuracy trends.

I also integrated Slack webhooks to send monitoring alerts. The Slack integration was tested successfully.

### CI/CD and Automation

The repository includes a GitHub Actions workflow for automated validation and evaluation. The pipeline covers dependency checks, baseline verification, monitoring tests, regression comparison, and artifact generation.

The project also includes support for prompt-change detection on trusted pull requests and an approval-controlled path for live API evaluation. This provides a foundation for safer prompt validation in team environments.

### Containerization

The evaluation runner is packaged in a Docker image based on Python 3.11. The runtime keeps the application environment reproducible and isolates dependency management from local machine state.

Secrets such as API keys are supplied at runtime rather than embedded in the image. Checkpoints are written to a mounted directory so evaluation outputs remain available after the container exits.

---

## Evaluation Results

### Docker Evaluation: Prompt v2 vs Dataset v3

I ran the complete 30-case evaluation inside a Docker container using the Groq API.

| Metric | Result |
|---|---:|
| Total test cases | 30 |
| Correct classifications | 29 |
| Incorrect classifications | 1 |
| Classification accuracy | **96.67%** |
| API or parsing errors | 0 |
| Execution environment | Docker / Python 3.11 |

All 30 test cases were processed successfully, and the evaluation results were saved to a JSON checkpoint.

One test case, `TC023`, was classified as `technical` instead of the expected `account` category.

### Baseline and Candidate Comparison

I compared the results of two prompt versions to identify changes in individual predictions.

Both versions achieved **96.67% classification accuracy**, but the regression comparator identified:

- **1 regression:** `TC023`
- **1 improvement:** `TC002`
- **Exact McNemar test p-value:** 1.0

The McNemar test did not show a statistically significant difference between the two versions.

Although both versions had the same overall accuracy, they made different predictions on individual cases. This is why I included case-level regression detection rather than relying only on the final accuracy score.

These results are based on a 30-case golden dataset and do not represent production-wide model performance.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM inference | Groq API |
| Model | GPT-OSS 20B |
| Prompt configuration | YAML |
| Evaluation datasets | JSON |
| Parallel execution | `asyncio` |
| Statistical analysis | Exact McNemar test |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Reporting | HTML and JSON |
| Monitoring | Python + Slack webhooks |
| Dashboard hosting | GitHub Pages |

---

## Repository Structure

```text
.
├── baselines/
├── config/
├── datasets/
├── docs/
├── evaluations/
├── policies/
├── prompts/
├── reports/
├── scripts/
├── src/
├── Dockerfile
├── README.md
├── requirements.txt
└── .github/
```

Key directories:

- `datasets/` contains versioned evaluation datasets
- `prompts/` stores prompt configurations for different versions
- `src/` includes the evaluation, scoring, regression, and reporting logic
- `evaluations/` stores checkpoints and run artifacts
- `reports/` and `docs/` contain generated dashboards and static documentation

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (for container-based execution)
- Groq API key

### Clone the repository

```bash
git clone https://github.com/Manvitha-kv352/Model-Regression-Detection-System.git
cd Model-Regression-Detection-System
```

### Local setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the evaluation:

```bash
python src/evaluation_runner.py
```

### Docker execution

Build the image:

```bash
docker build -t model-regression-detection:latest .
```

Create the output directory if it does not exist:

```powershell
New-Item -ItemType Directory -Force evaluations
```

Run the evaluation in Docker:

```powershell
docker run --rm --env-file .env -e EVALUATION_CHECKPOINT_PATH=evaluations/docker_v2_checkpoint.json -v "${PWD}/evaluations:/app/evaluations" model-regression-detection:latest
```

The checkpoint will be written to the local `evaluations` directory.

Do not commit `.env` files or API credentials to the repository.

---

## Reporting and Dashboard

The project includes a static HTML dashboard for exploring evaluation results and regression reports.

**[View Evaluation Dashboard](https://manvitha-kv352.github.io/Model-Regression-Detection-System/)**

The hosted dashboard reflects a published snapshot of results and is not yet fully automated to refresh after every evaluation run.

---

## Current Limitations

- The benchmark currently contains 30 evaluation cases.
- LLM outputs can vary between repeated runs.
- The hosted dashboard requires manual refreshes.
- Automatic pull-request evaluation still needs end-to-end validation.
- Quality-gate outcomes are not yet enforced as required merge checks.
- Drift monitoring requires a sufficient history of evaluations to be meaningful.

---

## Future Work

- complete end-to-end verification of automatic prompt-change evaluation
- enforce quality-gate decisions as required repository checks
- expand the golden dataset with more edge cases and ambiguous examples
- publish structured evaluation summaries directly to pull requests
- automate dashboard updates after each run
- extend Docker coverage to additional pipeline components

---

## Maintainer

**Manvitha KV**

B.Tech in Artificial Intelligence and Data Science

[GitHub](https://github.com/Manvitha-kv352)

---

This project was built to provide a practical and repeatable method for evaluating LLM behavior, validating prompt changes, and catching regressions before they affect real users.
