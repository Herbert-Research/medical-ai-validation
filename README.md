# Medical AI Validation Tools

Quantitative survival validation and probabilistic calibration workflows for AI-guided KLASS-standardized gastrectomy research.

## Why It Matters

Prospective surgical AI deployments rise or fall on whether model predictions translate into measurable survival gains. This notebooked toolkit links station-level metastasis risk models to disease-free survival endpoints, mirroring the validation pipeline I am developing for my PhD.

## Executive Snapshot

- Builds synthetic but clinically informed cohorts that contrast standard versus AI-guided gastrectomy.
- Quantifies survival benefit (Kaplan-Meier, log-rank, Cox proportional hazards) and probabilistic trustworthiness (Brier score, calibration slope/intercept, ECE).
- Benchmarks calibration remedies (raw model, Platt scaling, isotonic regression) to surface an evidence-based operating point.

## Repository Roadmap

- **Notebook**: `survival_and_calibration_enhanced.ipynb` · Sections mirror a real-world validation protocol.
- **Plots**: `kaplan_meier_example.png`, `calibration_curve_example.png` · Auto-generated evidence for quick review.
- **Script**: `demo_quickstart.py` · Headless quickstart that reproduces the notebook's core metrics.

## Getting Started

### 1. Reproducible Environment

The analyses were developed on Ubuntu 22.04, Python 3.11, 16 GB RAM. Estimated runtime < 1 minute.

```bash
# create environment (mamba or conda recommended)
mamba create -n med-ai-validation python=3.11
mamba activate med-ai-validation

# install runtime dependencies
pip install -r requirements.txt

# optional tooling for testing & CI linting
pip install -r requirements-dev.txt
```

### 2. Quickstart

```bash
# Smoke-test the end-to-end workflow
python demo_quickstart.py

# Explore the richer analysis
jupyter notebook survival_and_calibration_enhanced.ipynb

# Run lightweight regression tests
pytest test_demo_quickstart.py
```

## Analysis Highlights

- **Survival Outcomes** – Kaplan-Meier curves with 95 % CIs, log-rank hypothesis test, and Cox PH hazard ratios with confidence bounds.
- **Calibration Stress-Test** – Raw model, Platt-scaled, and isotonic-calibrated probability estimates with Brier score, calibration slope/intercept, and Expected Calibration Error (ECE).
- **Summary Dashboard** – Consolidated report of cohort statistics, discrimination (time-dependent concordance), and calibration quality flags aligned with surgical AI acceptance criteria.

![Kaplan-Meier survival comparison](kaplan_meier_example.png "Standard vs AI-guided gastrectomy survival curves with 95% confidence intervals")

![Calibration curve comparison](calibration_curve_example.png "Calibration curves contrasting raw model, sigmoid, and isotonic calibration")

## Key Metrics & Interpretation

- **Log-Rank p-value** < 0.05 indicates statistically significant survival divergence between arms.
- **Hazard Ratio (HR)** with 95 % CI grounds the effect size; HR < 1 favors AI-guided intervention.
- **Brier Score** quantifies probabilistic accuracy; values ≈0.05–0.10 reflect clinically actionable calibration.
- **Calibration Slope & Intercept** reveal under/over-confidence trends; post-calibration slope ≈1, intercept ≈0 is the target.
- **Expected Calibration Error (ECE)** ≤ 0.05 is the acceptance bar for deployment-readiness in the proposed validation protocol.

## Validation Workflow Blueprint

1. **Scenario Definition** – Construct station-specific metastasis risk predictions mapped to surgical interventions.
2. **Outcome Modeling** – Compare survival trajectories with non-parametric (Kaplan-Meier) and semi-parametric (Cox PH) approaches.
3. **Calibration Remediation** – Iterate on probabilistic outputs using Platt scaling and isotonic regression.
4. **Decision Readout** – Package metrics, plots, and a textual narrative for institutional review board (IRB) and multi-disciplinary tumor board stakeholders.

## Future Work

- Extend from synthetic to retrospective KLASS cohorts; integrate censoring patterns observed in Mannheim registry data.
- Bundle the workflow into a Snakemake/Prefect pipeline for prospective trial monitoring.
- Incorporate competing risk models and dynamic deep-learning-based survival estimators for station-level metastasis prediction.
- Add automated reporting (LaTeX/PDF) for regulatory submissions.

## Bibliography & Prior Work

- Dressler MH*, Choi JH*, et al. Bariatric Surgery in Patients With Prader-Willi Syndrome. *J Metab Bariatr Surg*. 2025;14(2):85-96. Demonstrated rigorous perioperative outcome tracking that informs this toolkit’s survival analyses. *Co-first author.

## Contact

Maximilian Herbert Dressler  
Mannheim Medical Faculty, University of Heidelberg  
Research Focus: Validation-first surgical AI for gastric cancer  
maximilian.dressler@medma.uni-heidelberg.de
```
