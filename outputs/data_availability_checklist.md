# Data Availability Checklist and Variable Mapping

This file is not a request that the hospital must provide all listed fields.

Its purpose is to help the modeling team inspect whatever data the hospital
actually provides, then decide:

- which variables are directly available;
- which variables can be mapped from hospital-specific field names;
- which variables may be extracted from reports or clinical notes by LLM/PLM;
- which variables are absent and should be dropped;
- which missing variables may limit the feasible modeling tasks.

## How to Use This Checklist

After receiving the hospital dataset, fill the following columns for each item:

```text
available_status:
  direct          = directly available as a structured field
  mapped          = available under a different hospital field name
  text_extract    = not structured, but may be extracted from reports/notes
  image_extract   = not structured, but may be inferred from image processing
  unavailable     = not available
  unknown         = not checked yet
```

Recommended working table:

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| age_at_diagnosis | baseline table | unknown |  | no | high | Basic predictor |
| BMI | baseline table | unknown |  | no | medium | May be missing |
| disease_duration | clinical note / baseline table | unknown |  | yes | high | Useful for prognosis |
| lesion_size | ultrasound report / image | unknown |  | yes | high | Can be extracted from report text if present |
| abscess_or_fluid_collection | ultrasound report / clinical note | unknown |  | yes | high | May affect treatment response |
| sinus_or_fistula | clinical note / ultrasound report | unknown |  | yes | high | Often clinically important |
| treatment_type | treatment record | unknown |  | partial | high | Required for treatment-effect questions |
| response_label | follow-up table / clinical note | unknown |  | partial | critical | Required as model target |

## 1. Patient Identity and Time Alignment

These fields are needed to connect different data files and prevent data
leakage.

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| anonymous_patient_id | all tables | unknown |  | no | critical | Must link table, reports, images, and follow-up |
| visit_or_encounter_id | HIS/EMR export | unknown |  | no | high | Helpful for longitudinal records |
| diagnosis_date | baseline table / note | unknown |  | partial | high | Needed for timeline |
| treatment_start_date | treatment record | unknown |  | partial | critical | Needed for prediction window |
| ultrasound_exam_date | ultrasound system/report | unknown |  | partial | high | Needed to align imaging before prediction |
| lab_test_date | lab table | unknown |  | no | high | Needed to distinguish baseline vs follow-up |
| follow_up_date | follow-up table/note | unknown |  | partial | critical | Needed for outcome definition |

## 2. Baseline Demographics and History

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| age_at_diagnosis | baseline table | unknown |  | no | high | Usually available |
| BMI | baseline table | unknown |  | no | medium | Height/weight can be mapped if BMI absent |
| smoking_status | baseline table / note | unknown |  | partial | medium | May be text-only |
| parity_or_birth_history | baseline table / note | unknown |  | partial | medium | Use if available |
| breastfeeding_history | baseline table / note | unknown |  | partial | medium | Helps distinguish non-puerperal status |
| menstrual_or_menopause_status | baseline table / note | unknown |  | partial | medium | Optional |
| prior_breast_disease | clinical history | unknown |  | yes | medium | Text extraction possible |
| prior_mastitis | clinical history | unknown |  | yes | medium | Recurrence-related context |
| autoimmune_disease_history | clinical history | unknown |  | yes | medium | Optional but clinically meaningful |
| tuberculosis_history_or_screening | clinical history/lab | unknown |  | partial | medium | Important for differential diagnosis |
| diabetes | baseline table / history | unknown |  | partial | medium | General risk factor |

## 3. Baseline Clinical Presentation

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| disease_subtype | diagnosis table / pathology | unknown |  | yes | high | Example: IGM, GLM, periductal mastitis |
| affected_side | note / ultrasound report | unknown |  | yes | medium | Left/right/bilateral |
| lesion_location | note / ultrasound report | unknown |  | yes | medium | Quadrant or clock-face position |
| disease_duration_before_treatment | note / baseline table | unknown |  | yes | high | Often not structured |
| breast_pain | note / assessment table | unknown |  | yes | medium | Can be binary or graded |
| palpable_mass | note | unknown |  | yes | medium | Usually text available |
| erythema_or_skin_redness | note | unknown |  | yes | medium | Inflammatory activity |
| swelling | note | unknown |  | yes | medium | Optional |
| nipple_discharge | note | unknown |  | yes | medium | Optional |
| ulceration_or_skin_breakdown | note | unknown |  | yes | high | May indicate severity |
| sinus_or_fistula | note / ultrasound report | unknown |  | yes | high | Important severity marker |
| fever_or_systemic_symptoms | note | unknown |  | yes | low | Use if available |
| recurrent_episode | history / follow-up | unknown |  | yes | high | May affect prognosis |

## 4. Laboratory Features

Laboratory indicators are useful only if their timestamps are available or can
be reliably aligned to treatment start.

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| white_blood_cell_count | lab table | unknown |  | no | medium | Baseline and longitudinal values useful |
| neutrophil_count_or_ratio | lab table | unknown |  | no | medium | Optional |
| lymphocyte_count_or_ratio | lab table | unknown |  | no | low | Optional |
| platelet_count | lab table | unknown |  | no | low | Optional |
| hemoglobin | lab table | unknown |  | no | low | Optional |
| CRP | lab table | unknown |  | no | medium | Inflammation marker |
| ESR | lab table | unknown |  | no | medium | Inflammation marker |
| liver_function | lab table | unknown |  | no | low | Treatment safety context |
| renal_function | lab table | unknown |  | no | low | Treatment safety context |
| fasting_glucose | lab table | unknown |  | no | low | Optional |
| prolactin_or_sex_hormones | lab table | unknown |  | no | low | Use if routinely available |
| microbiology_culture_result | lab/microbiology | unknown |  | partial | medium | May appear in notes |

## 5. Ultrasound Report and Image Features

These features may come from structured ultrasound tables, report text, or image
processing. If images are incomplete, report-derived variables may still be
valuable.

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| has_ultrasound_image | image folder / registry | unknown |  | no | critical | Needed for image branch availability |
| has_ultrasound_report | report table | unknown |  | no | high | Text branch availability |
| lesion_size_long_axis | ultrasound report / image | unknown |  | yes | high | Usually extractable from report |
| lesion_size_short_axis | ultrasound report / image | unknown |  | yes | medium | Optional |
| lesion_boundary | ultrasound report | unknown |  | yes | medium | Example: clear/unclear |
| lesion_shape | ultrasound report | unknown |  | yes | medium | Example: regular/irregular |
| echogenicity | ultrasound report | unknown |  | yes | medium | Example: hypoechoic/mixed |
| internal_liquefaction | ultrasound report | unknown |  | yes | high | Fluid/liquefaction clue |
| abscess_or_fluid_collection | ultrasound report / note | unknown |  | yes | high | Clinically important |
| sinus_or_fistula_sign | ultrasound report / note | unknown |  | yes | high | May be reportable |
| skin_thickening | ultrasound report | unknown |  | yes | medium | Optional |
| ductal_dilatation | ultrasound report | unknown |  | yes | medium | Relevant for periductal disease |
| vascularity_or_blood_flow | ultrasound report | unknown |  | yes | medium | Example: rich blood flow |
| multifocality | ultrasound report / image | unknown |  | yes | medium | Multiple lesions |
| axillary_lymph_node_status | ultrasound report | unknown |  | yes | low | Optional |
| radiologist_impression | ultrasound report | unknown |  | yes | medium | Free-text summary |

## 6. Pathology and Microbiology

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| pathology_available | pathology system | unknown |  | no | medium | Availability flag |
| pathology_date | pathology report | unknown |  | partial | medium | Time alignment |
| pathology_diagnosis | pathology report | unknown |  | yes | high | Diagnostic subtype |
| granulomatous_inflammation | pathology report | unknown |  | yes | medium | Text extraction possible |
| caseous_necrosis | pathology report | unknown |  | yes | medium | Differential diagnosis clue |
| plasma_cell_infiltration | pathology report | unknown |  | yes | medium | Relevant subtype feature |
| bacterial_culture | microbiology | unknown |  | partial | medium | If available |
| mycobacterium_or_fungal_test | microbiology/pathology | unknown |  | partial | medium | Differential diagnosis |

## 7. Treatment Variables

Treatment variables are essential if the project aims to compare continuing the
current treatment versus switching therapy.

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| current_treatment_type | treatment table / note | unknown |  | yes | critical | Required input |
| drug_name | medication table / note | unknown |  | partial | high | Standardize names |
| dose | medication table / note | unknown |  | partial | medium | Often messy |
| treatment_start_date | medication/treatment record | unknown |  | partial | critical | Timeline anchor |
| treatment_duration | derived from treatment dates | unknown |  | no | high | Derived variable |
| combination_therapy | treatment table / note | unknown |  | yes | high | Multiple therapies |
| antibiotics | medication record | unknown |  | partial | medium | Category flag |
| corticosteroids | medication record | unknown |  | partial | high | Category flag |
| methotrexate_or_immunosuppressants | medication record | unknown |  | partial | high | Category flag |
| aspiration_or_drainage | procedure record / note | unknown |  | yes | high | Procedure branch |
| surgery_or_excision | procedure record / note | unknown |  | yes | high | Procedure branch |
| treatment_adjustment | treatment record / note | unknown |  | yes | critical | Needed for switch-treatment question |
| adjustment_reason | clinical note | unknown |  | yes | high | Often text-only |
| adverse_event | note / medication record | unknown |  | yes | medium | Safety context |

## 8. Follow-up and Outcome Labels

Outcome labels determine what model can be built. If these are weak or missing,
the project may need to start from descriptive analysis or text structuring
instead of prediction.

| Standard variable | Possible source | Available status | Hospital field name | LLM/PLM extraction? | Importance | Notes |
|---|---|---|---|---|---|---|
| response_at_4w | follow-up table / note | unknown |  | partial | high | Use if follow-up schedule supports it |
| response_at_8w | follow-up table / note | unknown |  | partial | high | Candidate primary endpoint |
| response_at_12w | follow-up table / note | unknown |  | partial | high | Candidate primary endpoint |
| complete_response | follow-up note / clinician label | unknown |  | partial | high | Needs clinical definition |
| partial_response | follow-up note / clinician label | unknown |  | partial | high | Needs clinical definition |
| no_response_or_worse | follow-up note / clinician label | unknown |  | partial | high | Needs clinical definition |
| recurrence | follow-up note / registry | unknown |  | partial | high | Important secondary endpoint |
| treatment_escalation_required | treatment/follow-up record | unknown |  | partial | critical | Can be outcome or censoring event |
| lesion_size_change | ultrasound follow-up | unknown |  | yes | high | Derived from repeated ultrasound |
| pain_change | follow-up note | unknown |  | yes | medium | Text extraction possible |
| sinus_or_ulceration_change | follow-up note | unknown |  | yes | high | Severity response |

## 9. Feasibility Decision Rules

Use the completed checklist to decide the first modeling task:

| Available data pattern | Feasible first model |
|---|---|
| Structured baseline + clear outcome labels | Table-based baseline prediction |
| Reports/notes + clear outcome labels | Text-enhanced prediction or LLM extraction pipeline |
| Ultrasound images matched to patients + labels | Image-only or image+table model |
| Table + report + image matched by patient/time | Multimodal fusion model |
| Treatment changes + enough outcomes per treatment | Candidate treatment-response comparison |
| No reliable outcome labels | Data curation, cohort description, or endpoint construction only |

## 10. Immediate Review Questions After Data Arrival

- How many unique patients are available?
- How many patients have outcome labels?
- How many patients have ultrasound reports?
- How many patients have ultrasound images?
- How many patients have both image and structured table data?
- Can all records be aligned by anonymous patient ID?
- Are dates available for treatment start, ultrasound, labs, and follow-up?
- Are treatment changes recorded?
- Are outcome labels clinician-defined or must they be extracted from notes?

