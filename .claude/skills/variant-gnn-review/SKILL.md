---
name: variant-gnn-review
description: Use when reviewing, improving, documenting, testing, refactoring, or preparing the VARIANT-GNN repository for full compliance with the TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması şartnamesi, especially the Üniversite ve Üzeri genetic variant pathogenicity prediction task.
---

# TEKNOFEST 2026 Health AI Competition — Specification Compliance Skill

When this skill is active, audit the VARIANT-GNN repository strictly, technically, academically, and file-by-file against the TEKNOFEST 2026 Health AI Competition specification.

The goal is to bring the project into the highest possible level of compliance with the specification. Do not give general advice. Read the relevant files first, then make evidence-based assessments. Do not assume something exists if you have not seen it. If information is missing, explicitly state "no evidence found."

## 1. Official Competition Context

Evaluate the project against the following official context:

- Competition: TEKNOFEST 2026 Health AI Competition
- Category: University and Above
- Domain: Genetics
- Task: Classifying variants of unknown clinical significance as "Pathogenic" or "Benign"
- Variant type: Missense variants
- Reference classification: ACMG guidelines and criteria
- Ground truth: ACMG-compliant existing labels in source databases
- Primary success metric: F1 score
- Final score breakdown: Physical final / task performance 90%, final presentation 10%
- Code expectation: Must be runnable, reproducible, and clearly documented
- Jury authority: Finalist teams may be asked to re-run their code and reproduce declared results

Do not go beyond this context. The project must not be presented as a direct clinical diagnostic tool.

## 2. Dataset Requirements per Specification

Audit the project against the following dataset structure:

### Training Sets

1. General Dataset:
   - 1500 Pathogenic variants
   - 1500 Benign variants

2. Hereditary Cancer Panel:
   - 200 Pathogenic variants
   - 200 Benign variants

3. Phenylketonuria / PAH Gene Panel:
   - 200 Pathogenic variants
   - 200 Benign variants

4. Cystic Fibrosis / CFTR Gene Panel:
   - 70 Pathogenic variants
   - 70 Benign variants

### Test Sets

1. General Dataset:
   - 1000 Pathogenic variants
   - 1000 Benign variants

2. Hereditary Cancer Panel:
   - 100 Pathogenic variants
   - 100 Benign variants

3. Phenylketonuria / PAH Gene Panel:
   - 100 Pathogenic variants
   - 100 Benign variants

4. Cystic Fibrosis / CFTR Gene Panel:
   - 30 Pathogenic variants
   - 30 Benign variants

Note that the test set will be provided without labels during the competition. Flag as a critical violation any approach that behaves as if the repo has access to test labels, finds labels indirectly rather than predicting them, or extracts labels directly from external sources.

## 3. Class Definitions per Specification

Check the following class logic:

### Pathogenic Class

- Must originate from ClinVar and ClinGen.
- Variants reviewed by Expert Panel and Practice Guideline status must be considered.
- The context of 3- and 4-star reliability levels must be explained.
- "Pathogenic" and "Likely Pathogenic" must be merged under a single Pathogenic class.

### Benign Class

- The context of gnomAD healthy population variants must be explained in addition to ClinVar data.
- "Benign" and "Likely Benign" must be merged under a single Benign class.
- The purpose of reducing class imbalance must be stated.

If these class definitions are missing from the README, DATA_CARD, MODEL_CARD, or reports, record it as a critical documentation gap.

## 4. Genomic Address and Data Leakage Control

Per the specification, genomic address, chromosome, and position information is hidden to prevent competitors from directly finding labels from external data sources.

Therefore, perform the following checks:

- Does the code use genomic address, chromosome, or position information?
- Are labels being pulled from external databases?
- Is there a pipeline that can directly find class labels via ClinVar/ClinGen/gnomAD?
- Could a data enrichment process leak labels?
- Is preprocessing, scaling, imputation, feature selection, or SMOTE applied without a train/test split?
- Are SelectKBest, scaler, imputer, AutoEncoder, or graph construction fitted on the entire dataset?
- Does panel information create an indirect label signal for the model?
- Do the same or very similar variants appear in both train and validation/test splits?

If data leakage is suspected, treat it as a "Critical Issue."

## 5. Feature Groups per Specification

The repo, data description, and model architecture must explain the following feature groups:

1. Sequence and change information
   - Reference nucleotide
   - Alternative nucleotide
   - Codon change
   - Amino acid substitution

2. Local sequence and environmental context
   - 5 nucleotides before and after the variant
   - 5 amino acids before and after the relevant amino acid

3. Biochemical and structural effects
   - Hydrophobicity
   - Polarity
   - Molecular weight change
   - Possible effect on protein 3D structure

4. Evolutionary conservation
   - Phylogenetic diversity
   - Conservation across human populations
   - Conservation scores

5. Population data
   - Minor allele frequency
   - Population occurrence frequencies

6. In silico risk scores
   - Risk scores computed by different algorithms

For each feature group, audit the following:

- Is it explained in the Data Card?
- Is its intended use stated in the Model Card?
- Is it clearly described in the README?
- Does it have a preprocessing counterpart in the code?
- Is there a missing value strategy?
- Is the categorical/numerical conversion clear?
- Can it be produced in the competition test scenario?
- Does it carry a risk of label leakage?

## 6. Files and Folders to Inspect

First inspect the following files and folders:

- README.md
- DATA_CARD.md
- MODEL_CARD.md
- PROJECT_STATUS.md
- CHANGELOG.md
- CITATION.cff
- LICENSE
- SECURITY.md
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- pyproject.toml
- requirements.txt
- requirements-dev.txt
- requirements-ci.txt
- requirements-gpu.txt
- environment.yml
- environment-ci.yml
- environment-gpu-cu118.yml
- environment-gpu-cu121.yml
- Dockerfile
- Dockerfile.api
- docker-compose.yml
- Makefile
- app.py
- main.py
- trainer.py
- pipeline.py
- configs/
- src/
- tests/
- docs/
- reports/
- notebooks/
- models/
- data/
- data_contracts/
- .github/
- ci_pipeline_new.yml

If a file is missing, state "missing." If a file exists, evaluate it based on its content. Do not consider a file sufficient simply because its name exists.

## 7. README Compliance Check

The README must clearly and professionally include the following sections:

1. Project title
2. TEKNOFEST 2026 competition context
3. University-level genetic variant task
4. Pathogenic / Benign classification definition
5. Missense variant focus
6. Dataset structure
7. Train and test set separation
8. Genomic address masking rule
9. Feature groups used
10. Model architecture
11. Installation
12. Training command
13. Validation command
14. Inference / prediction command
15. Running with Docker
16. Running tests
17. F1 score calculation
18. Panel-based evaluation
19. Reproducibility explanation
20. Limitations
21. Clinical use disclaimer
22. Ethics and privacy statement
23. License
24. Citation information
25. Team / project information

List missing sections individually and provide a directly insertable correction for each.

## 8. MODEL_CARD Compliance Check

The MODEL_CARD must include the following:

- Model purpose
- Scope of use
- Situations where it must not be used
- Input format
- Output format
- Pathogenic / Benign class interpretation
- Model architecture
- Training data summary
- Validation method
- F1, precision, recall, confusion matrix
- Panel-based performance
- Calibration explanation
- Uncertainty estimation explanation
- Explainability method
- Data leakage prevention measures
- Clinical limitations
- Requirement for human expert oversight
- Ethics and privacy note
- Known issues
- Version information

Deduct points if missing.

## 9. DATA_CARD Compliance Check

The DATA_CARD must include the following:

- Data sources
- ClinVar description
- ClinGen description
- gnomAD description
- ACMG reference information
- Pathogenic class definition
- Benign class definition
- How "Likely" classes are merged
- Training set counts
- Test set counts
- Panel distributions
- Why genomic addresses are hidden
- Explanation of feature columns
- Information that column names will not be provided during the competition
- Missing values
- Class balance
- Bias risks
- Data leakage risks
- KVKK/GDPR compliance
- Secondary data use
- Research and education purpose
- Clinical use boundary

State explicitly if missing.

## 10. Model and Experiment Design Check

Inspect the following components file-by-file:

- GNN
- VariantGATv2GNN (GATv2Conv; eski alias VariantSAGEGNN — yalnız checkpoint uyumu)
- XGBoost
- LightGBM
- DNN
- Stacking ensemble
- Logistic regression meta learner
- AutoEncoder
- SelectKBest
- RobustScaler
- SMOTE
- k-NN graph construction
- Isotonic calibration
- MC Dropout
- SHAP
- GNNExplainer

For each component, state:

- Does it exist in the code?
- Is it explained in the README?
- Is it described in the MODEL_CARD?
- Is there an experimental result?
- Does it have a real contribution to the specification task?
- Does it create unnecessary complexity?
- Does it carry a data leakage risk?

## 11. F1 Score and Evaluation Check

Per the specification, F1 score is the primary metric in the final.

Check:

- Is F1 computed correctly?
- Is the TP, FP, FN logic correct for binary classification?
- Is accuracy over-emphasized?
- Are precision and recall reported separately?
- Is there a confusion matrix?
- Are the general dataset and the three panels evaluated separately?
- Is the validation result presented as if it were the final result?
- Is the final scenario with unknown test labels accounted for?
- Is the threshold selection explained?
- Is the change in F1 after calibration shown?

Flag any evaluation structure that is not F1-focused as a violation.

## 12. Reproducibility Check

When a jury member wants to run the project from scratch, the following steps must be clear:

1. Repository clone
2. Environment setup
3. Data placement
4. Config selection
5. Training
6. Validation
7. Prediction generation
8. F1 calculation
9. Test file creation
10. Running with Docker
11. Starting the API / interface
12. Verifying logs and outputs

Check the following:

- Is the Python version clearly stated?
- Are dependencies conflicting?
- Is the conda and pip path clear?
- Is the GPU/CPU distinction explained?
- Is the seed fixed?
- Is the config centralized?
- Are data paths hard-coded?
- Is there Windows/Linux compatibility?
- Is there sample data or a synthetic demo?
- Are model weights explained?
- Can the code produce a smoke test result with a single command?

If missing, record as a submission risk.

## 13. Test and CI Check

Check for the existence of the following tests:

- Unit test
- Preprocessing test
- Data schema test
- Leakage prevention test
- Train/validation split test
- F1 metric test
- Inference contract test
- Model smoke test
- API test
- Docker build test
- Config loading test
- Panel-based evaluation test
- Random seed determinism test

For missing tests, write the recommended file names:

- tests/test_data_schema.py
- tests/test_preprocessing_no_leakage.py
- tests/test_metrics_f1.py
- tests/test_inference_contract.py
- tests/test_panel_evaluation.py
- tests/test_reproducibility_seed.py
- tests/test_docker_smoke.py
- tests/test_config_loading.py

## 14. Clinical Safety, Ethics, and KVKK/GDPR Check

Per the specification, data used in the competition is publicly available, anonymized, and falls under secondary data use. Competition outputs cannot be used for clinical diagnosis, treatment, or medical decision support.

Therefore, check:

- Does the repo explicitly state "this is not a diagnostic tool"?
- Is there a statement that it is "for research and education purposes only"?
- Is the requirement for human expert oversight stated?
- Is KVKK/GDPR compliance proven and not overstated?
- Is the absence of PII explained in the correct context?
- Is the removal of genomic addresses explained as reducing re-identification risk?
- Is it stated that independent validation and regulatory compliance are required for clinical use?
- Are there any risky statements that directly recommend a decision to a patient or physician?

Replace risky statements with safe alternatives.

Example safe statement:

"This system cannot be used for clinical diagnosis, treatment, or independent medical decision-making. Model outputs must be interpreted solely within the scope of research, education, and competition evaluation. Independent validation, regulatory compliance, and expert physician assessment are required for any clinical use."

## 15. Report Compliance Check

Check the following specification expectations for PSR and PDR:

### Project Presentation Report

- General problem definition
- Literature review
- Proposed solution method
- Data approach
- Preliminary model / plan
- Clear connection to the specification task

### Project Detail Report

- Developed model architecture
- Training processes
- Internal test / validation results
- Evaluation methodology: F1 + MCC + PR-AUC mandatory per PDR template
- Code runnability
- Datasets used
- Result files
- F1-focused success analysis (PDR mandatory metrics: F1 + MCC + PR-AUC + Confusion Matrix)

If missing, propose how the report section should be written.

## 16. Scoring Rubric

Evaluate the project out of 100:

| Category | Maximum Score |
|---|---:|
| Specification Compliance | 20 |
| Scientific Validity | 20 |
| F1 and Evaluation Methodology | 15 |
| Data Leakage Prevention | 15 |
| Reproducibility | 10 |
| Code Quality | 10 |
| Documentation | 5 |
| Clinical Safety and Ethics | 5 |

Do not inflate scores. If there is no evidence, deduct points. Use the phrase "no evidence found in file."

## 17. Final Output Format

Always provide the answer in the following format:

## Executive Summary

Explain the project's specification compliance status in 5–8 sentences.

## Specification Compliance Matrix

| Specification Requirement | Repo Status | Evidence File | Risk | Correction |
|---|---|---|---|---|

## Critical Violations

For each item:

- Issue
- Affected file
- Specification reference
- Risk level
- Clear correction

## Moderate Violations

Use the same format for each item.

## Strengths

Only list strengths that have evidence in the files.

## Dataset and Label Compliance

Evaluate training/test sets, class definitions, panel structure, and ground truth logic.

## Data Leakage Analysis

Inspect genomic address, external data source, preprocessing, feature selection, SMOTE, graph construction, and validation risks.

## F1 Score and Final Evaluation Compliance

Evaluate F1 computation, TP/FP/FN logic, panel-based results, and the final scenario alignment.

## README Correction Plan

List missing README sections and propose directly insertable text.

## MODEL_CARD Correction Plan

List gaps and corrections.

## DATA_CARD Correction Plan

List gaps and corrections.

## Code and Pipeline Review

Provide file-based technical recommendations.

## Test and CI Review

List missing tests and recommended test files.

## Clinical Safety and Ethics Review

List risky statements and their safe alternatives.

## Reproducibility Review

Explain where a jury member would get stuck when running the project from scratch.

## Submission Readiness

Classify the project as one of the following:

- Ready
- Partially ready
- At risk
- Not ready

Provide justification.

## Prioritized Action Plan

### Must Be Fixed Within 24 Hours

### Must Be Fixed Before Submission

### Quality Improvements

## Final Score

| Category | Score | Maximum | Justification |
|---|---:|---:|---|

Provide the total score.

## Final Decision

Answer the following questions clearly:

1. Should the repo be submitted in its current state?
2. What are the 5 biggest gaps according to the specification?
3. What is the biggest scientific risk?
4. What is the biggest engineering risk?
5. What is the biggest documentation risk?
6. What are the first 10 tasks to approach a score of 100?

## 18. Writing Rules

- Write in English.
- Use formal and academic language.
- Evaluate with engineering discipline.
- Do not give general advice.
- Speak based on files.
- If there is no evidence, state "no evidence found."
- Make explicit connections to the specification.
- Do not use exaggerated praise.
- Do not use vague statements.
- Do not soften critical errors.
- Write an actionable solution for every problem.
