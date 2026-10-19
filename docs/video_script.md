# Demo video script (about 6 minutes)

*Glycaemic Digital Twin: Happiest Health Digital Twin Challenge 2026*

Recording setup: dashboard running (`scripts\demo.cmd` on Windows, `bash scripts/demo.sh` elsewhere) at
1440 x 900, browser zoom 100%. Have `docs/architecture.png`, `docs/figures/fig_auprc_ablation.png` and
`docs/figures/fig_ehr_gain.png` open in tabs. Keep the "synthetic data" banner visible whenever the
dashboard is on screen.

Every number below comes from `docs/ablation_results.md`. If you re-run the pipeline with a different
config, re-check them.

---

## 1. Problem (0:00-0:45)

**On screen:** title card, then a single CGM trace with a rice-meal spike and a pre-lunch low (dashboard timeline).

**Narration**
- India has about 101 million people with diabetes (ICMR-INDIAB, 2023), most of them Type 2.
- Control fails in two directions within hours: big post-meal spikes from carbohydrate-heavy meals, and
  dangerous lows from sulfonylureas and insulin, often after a skipped meal or at night.
- HbA1c every three months can't see either. We want a **2-hour early warning** for both, explained, with
  a way to ask "what if".

## 2. Data (0:45-1:40)

**On screen:** `docs/architecture.png` (columns 1-2), then `docs/synthetic_examples.png`.

**Narration**
- Two streams: a static EHR (demographics, HbA1c, eGFR, medications, comorbidities) and a 5-minute
  wearable stream (CGM, steps, heart rate, sleep).
- **Everything is synthetic.** 300 virtual patients, 14 days each, from a mechanistic simulator: Indian
  meals (rice, roti, idli, poha, sweets), insulin and sulfonylurea pharmacokinetics, poor-sleep insulin
  resistance, dawn and night effects, sensor noise and dropouts.
- Say this plainly: *EHR and sensor data never come from the same real people here. The link is synthetic,
  and we made EHR traits drive glucose on purpose so fusion has signal to find.*
- Calibrated rates: high in 27% of 2-hour windows, low in 4%, and metformin-only patients essentially never go low.

## 3. Model (1:40-2:40)

**On screen:** `docs/architecture.png` (columns 2-3), zoom on the split box and the model boxes.

**Narration**
- Every 30 minutes: take the last 6 hours plus the EHR profile, and predict "above 180" and "below 70" in the next 2 hours.
- Split **by patient**: 60 test patients never touched during training, model selection or alert thresholds.
- Ladder of models: persistence (extend the current trend) -> LightGBM on CGM features -> + wearables -> +
  EHR; and a PyTorch fusion net (EHR MLP + GRU, two heads, weighted loss for rare lows).
- One honest design choice: we score **onset** windows, where the patient isn't already high or low,
  because the rest are trivially easy.

## 4. Results (2:40-3:40)

**On screen:** `docs/figures/fig_auprc_ablation.png`, then `docs/figures/fig_ehr_gain.png`.

**Narration**
- Headline metric for rare lows is AUPRC (chance = 0.02). Persistence: 0.19. LightGBM on CGM: 0.65.
  Adding the EHR: **0.71**. Hyperglycaemia: 0.38 -> 0.72 -> 0.74.
- Does the EHR add value? On synthetic data, yes: **+0.05 for lows in LightGBM, CI clear of zero**, but the
  simulator builds that in. So we tested it on **real linked records** (ShanghaiT2DM, 100 patients): there
  the EHR added **nothing** to a 2-hour CGM forecast, while the CGM model predicted real lows well (AUROC
  0.89). We changed our claim to match. Wearables added nothing measurable.
- Operating point: at a 10% false-alert rate we catch 98% of upcoming lows, but only about 1 in 5 alerts
  is a true low. Alert fatigue is a real open problem.
- We deploy LightGBM + EHR because it's the best for lows and its probabilities are well calibrated.

## 5. Live dashboard demo (3:40-5:30)

**Scene A: a dangerous low (P0014, metformin + sulfonylurea, HbA1c 7.1%)**
1. Sidebar: patient **P0014**, prediction time **Highest hypoglycaemia risk** (Sun 18 Jan, 09:55).
2. Point at the risk cards: **hypo 95%, ALERT**; hyper 0%.
3. **Glucose timeline:** overnight rise toward 200, morning fall. The hypo risk line climbs before glucose
   crosses 70, and the dotted "actual next 2 h" trace confirms the low.
4. **Alert drivers:** current glucose 70 mg/dL, the sulfonylurea dose (EHR), time of day. *"These explain the
   model, not the physiology."*
5. **What-if -> Extra insulin dose:** hypo risk rises to 99%; the simulator shows the low deepening (56 -> 48 mg/dL).
6. **What-if -> Skipped meal** (if a meal is in the window): risk and simulated minimum move together.

**Scene B: late dinner (P0184, metformin + sulfonylurea)**
1. Untick "Only patients with hypoglycaemia events" if needed, pick **P0184**, then **Choose a time ->
   Fri 16 Jan -> 20:25**.
2. **What-if -> Late dinner (+2 h):** the yellow note says the model **cannot see** this. The dinner hasn't
   happened yet, so model risk is unchanged. Meanwhile the simulator shows glucose falling (next-2 h minimum
   ~245 -> ~170 mg/dL) because the sulfonylurea acts without food.
3. Takeaway: logging planned meals and doses is the obvious next input for a real twin.

**Scene C (optional): Patient overview**
- Profile, labs, medication, ICD-10 codes, 14-day time in range / below / above, GMI and alert counts.

## 6. Limitations and next steps (5:30-6:10)

**On screen:** README "Limitations" section.

**Narration**
- Synthetic data, synthetic linkage, built-in EHR signal: these results don't measure clinical value.
- The simulator's lows drift in slowly, so hypo performance and lead time are optimistic.
- Hypo results rest on 17 test patients (7 account for 80% of lows), so the intervals are wide.
- Next: validate on real paired data (e.g. a prospective CGM + EHR cohort), add meal and dose logging,
  per-episode alerting with lower alert burden, subgroup and fairness checks, clinical review.

## 7. Close (6:10-6:20)

**On screen:** dashboard banner + repo URL.

**Narration:** "Open source under MIT, one command to rebuild everything. Synthetic data, not for clinical use. Thank you."
