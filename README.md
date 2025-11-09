# Medical AI Validation Tools: Survival & Calibration

Supporting analytics package for the proposed PhD dissertation **“Prospective Validation of Station-Specific Risk Guidance for KLASS-Standardized Gastrectomy.”** This repository establishes the quantitative framework for linking station-level machine learning predictions to disease-free survival (DFS) endpoints, a critical prerequisite for Aim 1 clinical deployment.

## Executive Summary

  - **Synthesizes** clinically informed cohorts to model the actuarial impact of AI-guided lymphadenectomy versus standard KLASS protocols.
  - **Benchmarks** probabilistic trustworthiness using Brier scores and Expected Calibration Error (ECE), comparing raw model outputs against Platt scaling and isotonic regression remedies.
  - **Pilots** external validation workflows using TCGA STAD data to ensure the pipeline is ready for forthcoming institutional datasets.
  - **Exports** audited figures and CSV summaries designed for institutional review board (IRB) and multi-disciplinary tumor board review packets.

## Scientific Context and Objectives

Before surgical AI models can be deployed intraoperatively, their outputs must be rigorously calibrated to match actual recurrence risks. This codebase substantiates the translational premise that improvements in Area Under the Curve (AUC) are insufficient for clinical adoption without demonstrable calibration quality and linkage to survival outcomes. The included workflows provide the "validation-first" evidence required to transition from retrospective modeling to prospective interventional trials.

## Data Provenance and Governance

  - **Synthetic Stream:** Primary validation uses controlled synthetic cohorts (`n=240`, `seed=42`) encoded with established clinical priors (e.g., tumor stage hazards derived from KLASS-02 literature) to stress-test statistical machinery without PII concerns.
  - **Clinical Pilot Stream:** Secondary validation ingests TCGA PanCanAtlas 2018 data (`data/tcga_2018_clinical_data.tsv`) when available, adhering to strict de-identification standards.
  - **Compliance:** All analytical outputs remain de-identified. Synthetic data is algorithmically generated and contains no protected health information (PHI).

## Analytical Workflow

The core pipeline (`demo_quickstart.py`) executes four phases:

1.  **Scenario Definition** – Generates balanced cohorts contrasting standard resection against AI-guided intervention based on station-specific risk profiles.
2.  **Outcome Modeling** – Quantifies survival divergence using non-parametric Kaplan-Meier estimators and semi-parametric Cox Proportional Hazards models (reporting Hazard Ratios with 95% CIs).
3.  **Calibration Stress-Test** – Evaluates the trustworthiness of risk predictions via reliability diagrams, calculating calibration slope/intercept and ECE to determine if post-hoc remediation (Sigmoid vs. Isotonic) is necessary for deployment.
4.  **Decision Readout** – Consolidates metrics into standardized CSVs and high-resolution PNGs for stakeholder review.

## Example Output

Running the default configuration produces clinically-interpretable summaries demonstrating the pipeline's analytical rigor:

=== Survival Analysis Summary ===
Median DFS (AI-guided): 28.5 months
Median DFS (Standard):  10.5 months
Log-rank p-value:      0.0000
Hazard ratio (AI-guided vs Standard): 0.52 (95% CI 0.38–0.70)
C-index:                0.698
PH global p-value:      0.760

=== Calibration Summary ===
                           Model  Brier Score  ECE (10-bin)  Calibration Slope  Calibration Intercept
Gradient Boosting (uncalibrated)        0.144         0.054              0.615
  -0.499
         Platt Scaling (sigmoid)        0.143         0.076              3.524
   3.702
             Isotonic Regression        0.140         0.055              1.557
   0.779


=== TCGA-STAD Pilot (Node-positive vs Node-negative) ===
Median PFS (Node-positive): 25.5 months | Median PFS (Node-negative): inf months
Log-rank p-value: 0.0012 | PH global p-value: 0.396
Hazard ratio: 1.88 (95% CI 1.24–2.85)

Group-level summary (TCGA):
        group  patients  median_months  event_rate
Node-negative       121           14.4       0.248
Node-positive       273           12.4       0.396

This output quantifies both the potential clinical benefit (11-month DFS improvement) and model trustworthiness (ECE < 0.06 threshold) required for translational deployment.

## Generated Figures

**Survival Analysis Output**

  - `kaplan_meier_example.png` – Comparative DFS curves (Standard vs. AI-guided) with 95% confidence intervals, visualizing the potential actuarial gain of targeted guidance.
  - `tcga_kaplan_meier.png` – (If data present) Real-world anchor showing Progression-Free Survival stratified by N-stage.

**Calibration Output**

  - `calibration_curve_example.png` – Reliability diagram contrasting uncalibrated model confidence against observed metastasis frequencies, overlaying Platt and Isotonic corrections to guide operating point selection.

## Generated Tables

All quantitative outputs are exported as machine-readable CSVs for audit trails and downstream statistical reporting:

**Survival Analysis Tables**

  - `synthetic_survival_summary.csv` – Cox model coefficients, hazard ratios with confidence intervals, concordance indices, and log-rank test statistics for the simulated cohort comparison.
  - `tcga_survival_summary.csv` – (If TCGA data present) Analogous survival metrics computed from real-world STAD patient data.
  - `tcga_group_summary.csv` – (If TCGA data present) Patient counts, median survival times, and event rates stratified by clinical subgroup.

**Calibration Tables**

  - `calibration_summary.csv` – Comparative performance metrics (Brier Score, ECE, calibration slope/intercept) across uncalibrated baseline and post-hoc correction methods.

These structured outputs enable direct import into institutional REDCap databases or statistical verification packages (SAS, Stata, R) as required by publication checklists.

## Usage

### Quickstart (Headless)

Run the end-to-end validation pipeline using standard defaults.

```bash
# Setup environment
pip install -r requirements.txt

# Execute full workflow
python demo_quickstart.py --output-dir reports/
```

### Interactive Analysis

Launch the accompanying notebook for a narrative walkthrough of the methodology, suitable for faculty review.

```bash
jupyter notebook survival_and_calibration_enhanced.ipynb
```

### Configuration Options

The workflow can be parameterized to stress-test different clinical scenarios or validate against alternative datasets.

```bash
# Increase synthetic cohort size for higher statistical power simulations
python demo_quickstart.py --n-patients 1000 --metastasis-samples 5000

# Point to an alternative clinical TSV for external piloting
python demo_quickstart.py --tcga-path /path/to/local_registry_data.tsv
```

## Software Requirements

  - Python 3.9 or newer.
  - Core dependencies: `pandas`, `numpy`, `scikit-learn`, `lifelines` (for survival statistics), `matplotlib`/`seaborn` (for visualization).
  - See `requirements.txt` for pinned versions used in validation.

## Input Validation Schema

When ingesting external clinical data (e.g., TCGA), the pipeline enforces strict schema validation to ensure integrity. The script halts if these expected AJCC-concordant fields are missing:

  - `Progress Free Survival (Months)` / `Disease Free (Months)` → coalesced to `time_months`
  - `Progression Free Status` / `Disease Free Status` → parsed to numerical `event` flags
  - `American Joint Committee on Cancer Tumor Stage Code` → mapped to numeric `tumor_stage`

This safeguard prevents silent failures when reporting outcomes to screening committees.

## Clinical Interpretation Notes

  - **Synthetic nature:** The primary survival benefits reported in `kaplan_meier_example.png` are derived from clinically-informed simulations to demonstrate pipeline readiness; they do not yet represent results from a prospective trial.
  - **TCGA Proxy:** In the external pilot, N-stage positivity is used as a proxy for "high-risk" status to validate the Cox PH machinery on real-world heterogeneous data.
  - **Calibration threshold:** An Expected Calibration Error (ECE) \> 0.05 is currently flagged as requiring remediation before model outputs should be displayed to surgeons.
  - **Undefined Medians:** In low-risk cohorts (e.g., TCGA Node-negative), median survival may be reported as `inf` (infinite) if fewer than 50% of patients experience an event during the study period. This reflects robust handling of right-censored data rather than an error.

## Repository Stewardship

Author: **Maximilian Herbert Dressler**

## Acknowledgement

“The results presented here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.”

## Citations

  - Cerami E, Gao J, Dogrusoz U, *et al.* The cBio Cancer Genomics Portal: An Open  Platform for Exploring Multidimensional Cancer Genomics Data. *Cancer Discovery.* 2012;2(5):401–404.
  - Gao J, Aksoy BA, Dogrusoz U, *et al.* Integrative Analysis of Complex Cancer Genomics and Clinical Profiles Using the cBioPortal. *Science Signaling.* 2013;6(269):pl1.
  - Liu J, Lichtenberg T, Hoadley KA, *et al.* An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. *Cell.* 2018;173(2):400–416.e11.