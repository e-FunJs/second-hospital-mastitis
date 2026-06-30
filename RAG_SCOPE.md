# RAG Scope: Non-puerperal Mastitis Treatment Response

## 1. Purpose

This RAG knowledge base supports an AI project for predicting treatment response
in non-puerperal mastitis. It provides traceable external medical knowledge for
feature design, endpoint definition, clinical text structuring, and explanation
generation.

The RAG library is not a replacement for hospital patient-level data. Final
prediction models must be trained and validated on local clinical data with
clear treatment timelines and follow-up outcomes.

## 2. Target Clinical Questions

Primary questions:

- What baseline and longitudinal factors are associated with treatment response?
- Which ultrasound features are relevant to disease status and treatment outcome?
- What treatment options are reported for non-puerperal mastitis and related
  subtypes?
- How do studies define response, remission, recurrence, treatment failure, and
  escalation?

Future modeling questions:

- What is the expected outcome if a patient continues the current treatment?
- What is the expected outcome if a patient switches to a candidate therapy?

The second question requires treatment-effect modeling and prospective clinical
validation. Literature knowledge can support hypothesis generation, but cannot
alone establish individualized treatment recommendations.

## 3. Disease Scope

Include literature related to:

- non-puerperal mastitis;
- nonlactational mastitis;
- idiopathic granulomatous mastitis;
- granulomatous lobular mastitis;
- periductal mastitis;
- plasma cell mastitis;
- breast abscess when relevant to non-lactational inflammatory breast disease.

Chinese terms:

- 非哺乳期乳腺炎;
- 肉芽肿性乳腺炎;
- 特发性肉芽肿性乳腺炎;
- 肉芽肿性小叶性乳腺炎;
- 浆细胞性乳腺炎;
- 乳腺脓肿;
- 乳腺炎 复发;
- 乳腺炎 超声.

## 4. Evidence Priorities

Highest priority:

1. Guidelines, consensus statements, and expert recommendations.
2. Systematic reviews and meta-analyses.
3. Prospective or retrospective clinical cohorts with treatment outcomes.
4. Imaging studies describing ultrasound findings and follow-up changes.
5. Treatment-specific studies, including corticosteroids, antibiotics,
   methotrexate, drainage, surgery, and combined treatment.

Lower priority:

- small case series;
- single case reports;
- general mastitis papers without relevance to non-puerperal disease;
- papers without outcomes, treatment, imaging, or clinically useful variables.

## 5. Inclusion Criteria

Include a paper if it contains at least one of the following:

- disease definition or subtype classification;
- treatment strategy;
- treatment outcome or recurrence;
- predictors of response or recurrence;
- ultrasound, pathology, or laboratory features;
- follow-up schedule or endpoint definition.

## 6. Exclusion Criteria

Exclude or down-rank:

- lactational/puerperal mastitis only;
- pediatric or male-only cases unless mechanistically relevant;
- animal-only or in-vitro-only studies;
- papers without accessible title/abstract metadata;
- inaccessible copyrighted full text unless only metadata/abstract is used.

## 7. Planned RAG Use Cases

### 7.1 Candidate Feature Design

Use retrieved evidence to propose hospital data fields:

- demographics and risk factors;
- disease presentation;
- ultrasound features;
- laboratory results;
- treatment details;
- follow-up outcomes.

### 7.2 Text Structuring

Use the knowledge base to guide extraction from:

- ultrasound reports;
- diagnosis notes;
- treatment progress notes;
- follow-up records.

### 7.3 Model Explanation

Use retrieved citations to explain why certain variables may be clinically
relevant. The prediction itself should come from validated models, not from
unconstrained LLM judgment.

## 8. Data Governance

Keep public literature and hospital patient data separated:

- public RAG library: open literature, guidelines, abstracts, open-access full
  text;
- clinical RAG or patient context: de-identified hospital records only, with
  restricted local access.

Do not mix identifiable patient records into the public literature knowledge
base.

## 9. Day 1 Completion Criteria

Day 1 is complete when the repository contains:

- this scope document;
- source and query configuration files;
- an initial metadata schema;
- a minimal PubMed search script;
- candidate variable and endpoint templates.

