# Path to real-world deployment (India)

**This is a plan, not a claim.** The prototype runs on synthetic data and is not a medical device. This page
sets out what would have to happen before a system like it could support real clinicians in India. It is
not legal or regulatory advice: anything marked *to confirm* needs checking against the primary text or with
the regulator.

## 1. Evidence on real patients

| Step | What | Done when |
|---|---|---|
| 1a | **Retrospective validation** on real, linked EHR + CGM records from Indian T2D clinics (insulin and sulfonylurea users included), with ethics committee approval following ICMR's AI guidelines | Discrimination and calibration reported overall and per subgroup, with patient-level CIs |
| 1b | **Recalibration** on that population. Our zero-shot model was miscalibrated on real US CGM (ECE 0.14); leave-one-participant-out Platt scaling brought it to 0.024 without retraining ([external_validation.md](external_validation.md)) | Calibration error within an agreed bound |
| 1c | **Silent prospective pilot**: the model scores in the background and clinicians never see the alerts; outcomes are compared afterwards | Episode-level sensitivity, false alerts per patient-day and lead time measured prospectively |
| 1d | **Clinician-in-the-loop pilot** with diabetologists, judged on alert usefulness, workload and safety events | Agreed safety and usefulness thresholds met |

What exists today: a synthetic proof of concept, plus a real-CGM sanity check on 14 US participants
([external_validation.md](external_validation.md)).

## 2. Data and consent

* **DPDP Act 2023 and DPDP Rules 2025.** The Rules were notified in November 2025 with a staggered roll-out:
  the consent, notice and security obligations apply 18 months after notification, and Consent Manager
  registration one year after ([Hogan Lovells](https://ca.hoganlovells.com/en/publications/indias-digital-personal-data-protection-act-2023-brought-into-force-),
  [India Briefing](https://www.india-briefing.com/news/dpdp-rules-2025-india-data-protection-law-compliance-40769.html)).
  For this system that means:
  * free, specific, informed consent for combining EHR and wearable data, given as a standalone notice
    and as easy to withdraw as to give;
  * purpose limitation (glucose risk support only) and data minimisation;
  * the Rule 6 security safeguards, including encryption, access control, masking, and access logs kept for a year;
  * breach notification to affected people and to the Data Protection Board, the latter within 72 hours.
  * How health data interacts with sector-specific rules: *to confirm* with health-sector legal review.
* **Linkage.** Real linkage would replace this project's synthetic enrollment table. It would use the
  patient's consented link between their clinic record and their device account, ideally via their ABHA
  (Ayushman Bharat Health Account) and ABDM consent flows, rather than any matching by attributes.
* **Interoperability.** Exchange EHR data and CGM readings as FHIR resources through ABDM-compliant
  systems (*to confirm*: the applicable NRCeS FHIR profiles and the CGM device-integration route).

## 3. Regulation

* Software with a medical purpose falls under the **Medical Devices Rules 2017**. CDSCO's
  [draft guidance on medical device software](https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2018/UploadPublic_NoticesFiles/Draft%20guidance%20document%20on%20Medical%20Device%20Software%2021%2010%202025.pdf)
  (21 October 2025, draft) distinguishes standalone software (SaMD) from embedded software.
  * Risk class (A-D) depends on the intended purpose, how much the output drives clinical decisions, and
    the severity of the condition ([India Briefing](https://india-briefing.com/news/cdsco-draft-guidance-medical-software-40691.html)).
  * A tool that warns of hypoglycaemia and informs treatment is unlikely to be the lowest class. The actual
    class is *to confirm* with a regulatory assessment.
* **Ethics.** Follow ICMR's
  [Ethical Guidelines for Application of AI in Biomedical Research and Healthcare (2023)](https://www.icmr.gov.in/ethical-guidelines-for-application-of-artificial-intelligence-in-biomedical-research-and-healthcare):
  ethics committee review, informed consent, transparency, accountability and bias assessment.

## 4. Product and safety

* **A supplement, never a replacement:** the CGM's own low-glucose alarms stay on, and the twin's alerts never suppress them.
* **Per-episode alerting** with a refractory period and a daily alert budget, re-tuned with clinicians on real
  episodes (the synthetic-data policy and its trade-off curve are in [alert_policy.md](alert_policy.md)).
* **Inputs a real twin needs:** logged meals and doses (the model can't see a planned dinner or a fresh
  dose today), device wear status, and medication changes from the EHR.
* **Personalisation:** fit each patient's twin parameters from their own first weeks of data
  ([twin_personalisation.md](twin_personalisation.md)), and re-fit when medication changes.
* **Explanations** that clinicians have reviewed for usefulness, with misleading drivers flagged.

## 5. Operations

* Monitoring of data quality (CGM coverage, sensor changes), score drift and calibration, with
  recalibration triggers.
* Versioned models with the evidence behind each version recorded (this repo's model card and ablation
  reports are the template).
* Incident reporting for missed lows and harmful alerts, feeding back into evaluation.
* A subgroup audit each release, adding region, diet, socioeconomic status and device type, which the
  synthetic data lacks.
