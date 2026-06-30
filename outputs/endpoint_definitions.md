# Endpoint Definitions Draft

This document is a discussion draft for aligning the clinical team and modeling
team. Final endpoints must be confirmed by clinicians before modeling.

## 1. Primary Prediction Target

Recommended first primary target:

```text
Effective response at a fixed follow-up window after treatment start or treatment adjustment.
```

Suggested windows:

- 4 weeks;
- 8 weeks;
- 12 weeks.

The exact window should match the hospital's real follow-up schedule.

## 2. Binary Response Endpoint

Example:

```text
effective = complete_response OR partial_response
ineffective = stable_disease OR worsening OR treatment_escalation_required
```

This is easier for the first model, but may lose clinical detail.

## 3. Ordinal Response Endpoint

Example categories:

1. complete_response
2. partial_response
3. stable_or_no_clear_response
4. worsening
5. recurrence

This preserves more clinical information but requires more samples and more
consistent labeling.

## 4. Candidate Clinical Definitions

### Complete Response

All or nearly all of the following:

- symptoms resolved;
- mass disappeared or nearly disappeared;
- ultrasound lesion resolved or minimal residual change;
- no drainage, ulceration, or sinus;
- no treatment escalation required.

### Partial Response

One or more clear improvements:

- pain improved;
- mass size reduced;
- ultrasound lesion size reduced;
- abscess/fluid collection reduced;
- inflammatory markers improved;
- current treatment continued without escalation.

### Ineffective or Worsened

One or more of the following:

- symptoms not improved;
- lesion enlarged;
- new abscess, sinus, or ulceration;
- inflammatory markers not improved when clinically relevant;
- treatment needed escalation or switching because of poor response.

### Recurrence

Disease reappears after a documented remission or clear response. Record:

- recurrence_date;
- recurrence_site;
- recurrence_after_complete_response_or_partial_response;
- treatment_required_for_recurrence.

## 5. Treatment Strategy Targets

For future individualized treatment-effect modeling:

```text
P(response | current patient state, treatment = continue_current)
P(response | current patient state, treatment = switch_to_candidate_A)
P(response | current patient state, treatment = switch_to_candidate_B)
```

This requires enough historical patients for each treatment strategy and careful
control of confounding. It should be treated as a later-stage model.

## 6. Time Alignment Rules

To reduce data leakage:

- baseline predictors must occur before treatment start;
- dynamic predictors must occur before the prediction time;
- outcome text after the target follow-up window must not be used as input;
- treatment changes after the prediction time must be treated as outcomes or
  censoring events, not baseline predictors.

## 7. Endpoint Questions for Clinicians

- What is the routine follow-up schedule?
- Is 4, 8, or 12 weeks the most meaningful first response window?
- What criteria define effective treatment in local practice?
- Is treatment escalation itself an "ineffective" endpoint?
- How should spontaneous symptom fluctuation be handled?
- How should incomplete follow-up be handled?

