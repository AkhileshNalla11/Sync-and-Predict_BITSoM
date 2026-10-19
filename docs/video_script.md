# Video script (about 25 minutes)

*Glycaemic Digital Twin: Happiest Health Digital Twin Challenge 2026*

The challenge asks for a video of at least 20 minutes (a PowerPoint recording or an unlisted YouTube
link). This script follows [presentation.pptx](presentation.pptx) slide by slide, with one live
dashboard demo in the middle. Timings add up to about 25 minutes at a calm pace (about 130 words per
minute), which leaves room to slow down without dropping under 20.

## Before recording

* **Slides:** `docs/presentation.pptx`. Fill in the team name and college on slides 1 and 20 first. The
  speaker notes on each slide are a short version of the narration below.
* **Dashboard:** start it with `scripts\demo.cmd` (Windows) or `bash scripts/demo.sh`, or use the live
  Streamlit Cloud link. Browser at 1440 x 900, zoom 100%. Select patient **P0014** and prediction time
  **Highest hypoglycaemia risk** before you start. Keep the "synthetic data" banner visible whenever the
  dashboard is on screen.
* **Recording:** PowerPoint's *Record* (Slide Show > Record) captures slides, voice and an optional camera,
  and exports to MP4 (File > Export > Create a Video). For the demo, record the browser with the Windows
  Game Bar (Win + Alt + R) or OBS, then join the clips. Upload as **unlisted** on YouTube and put the link in
  the README.
* **Numbers:** every figure below comes from the docs in this folder: [model_card.md](model_card.md),
  [alert_policy.md](alert_policy.md), [twin_personalisation.md](twin_personalisation.md),
  [twin_intervals.md](twin_intervals.md), [external_validation.md](external_validation.md) and
  [external_shanghai.md](external_shanghai.md). If you re-run anything, re-check them.
* **Two rules for the whole video:** say "synthetic" whenever a result comes from synthetic patients, and
  never say or imply the system is clinically validated.

---

## 1. Opening (0:00-1:00) · slide 1

**Narration**

Hello, we are *[team name]* from *[college or incubator]*. Our project for the Happiest Health Digital
Twin Challenge is a glycaemic digital twin for people with Type 2 diabetes.

In one sentence: we built a computer copy of a patient that combines their health record with their
wearable data, predicts dangerous high and low glucose two hours before it happens, explains every
warning, and lets a doctor test "what if" on the twin before trying it on the patient.

Then we did the part we think matters most. We checked every claim on real patient data, and when real
data disagreed with us, we changed our claims. You will see exactly where that happened.

Everything you will see is a research prototype. It is trained on synthetic data and it is not for
clinical use.

## 2. The problem (1:00-2:30) · slide 2

**Narration**

India has about 101 million people with diabetes, according to the ICMR-INDIAB study published in 2023,
and most of them have Type 2.

Day to day, glucose control fails in two opposite directions. The first is the spike after a meal.
Traditional Indian meals are rich in carbohydrates: rice, roti, idli, poha. With insulin resistance they
turn into large, long spikes above 180 milligrams per decilitre, and those repeated spikes drive the
long-term complications of diabetes.

The second is the low, below 70. Sulfonylureas are widely prescribed in India because they are cheap and
work well, and together with insulin they can push glucose too low, especially after a skipped or late
meal, extra activity, or an extra dose. The worst case is at night, when the patient is asleep and
cannot react.

The doctor sees an HbA1c every few months. These events happen within hours. The information to see
them coming exists, but it lives in two places: the record, with diagnoses, labs and medicines, and the
sensor, with a glucose reading every five minutes. Nobody combines them into an early warning a doctor
can act on. That is the gap we set out to close.

## 3. Our answer (2:30-3:30) · slide 3

**Narration**

Our answer is a digital twin of each patient, with four capabilities.

First, predict. Every 30 minutes, from the last six hours of wearable data plus the record, the twin
estimates the chance of glucose going above 180 or below 70 in the next two hours.

Second, alert. A risk score alone would fire constantly, so an alert policy raises one alert per risk
episode. That works out to about one low-glucose alert every two days per patient.

Third, explain. Every risk comes with its main reasons, for example "glucose is falling" or "this
patient takes a sulfonylurea".

Fourth, simulate. The twin is fitted to the patient's own data, so a clinician can ask what happens with
a late dinner, a skipped meal, a poor night's sleep or an extra insulin dose, and see the answer with an
uncertainty band.

## 4. Architecture (3:30-5:30) · slide 4

**Narration**

Here is how it fits together, in five layers from left to right.

Layer one is the data. A synthetic EHR cohort of 300 patients feeds a physiology simulator, which
produces the wearable stream: CGM every five minutes with realistic sensor lag, noise and dropouts, plus
steps, heart rate and sleep. An enrollment log links each patient to a device, the way a real study
would.

Layer two links the streams and builds features. A linker joins records to devices and checks the join.
Rolling windows take six hours of history and label the next two hours. The record becomes a static
vector, scaled using training patients only. Then we split by patient, never by time.

Layer three is the models: a simple trend baseline, LightGBM with one model per event, and a deep fusion
network with a GRU for comparison.

Layer four is evaluation, and the teal box is the important one: real external data, used only to check
us, never to train.

Layer five is the twin itself: the patient state, the explanations, the personalised what-if twin, and
the Streamlit dashboard a clinician uses.

*(Optional: switch to the GitHub repository for 30 seconds.)* In the repository, each layer is a folder.
`src/data` holds the simulator, the linker and the feature builders. `src/models` holds the baselines and
the fusion network. `src/eval` holds every evaluation, including the two real-data checks. `src/twin` holds
the patient state, explanations, the what-if engine and the personalisation, and `src/app` is the dashboard.
Every result in this video has a matching report in the `docs` folder, generated by the code rather than
typed by hand.

## 5. Data, honestly labelled (5:30-7:30) · slide 5

**Narration**

We need to be upfront about the data. No public dataset links hospital records and wearable data for
the same Indian patients. So we built a physiology simulator: a Bergman-style glucose model with Indian
meals, drug pharmacokinetics for metformin, sulfonylureas and three kinds of insulin, sleep, activity
and the morning rise in glucose. It generates 300 patients, 14 days each.

A few details make it more than a toy. Doctors titrate doses, so patients with worse control are on
stronger drugs, as in real clinics. The CGM is not the true glucose: it lags blood glucose by about ten
minutes, has correlated noise, and drops out for stretches, like a real sensor. Wearables come with a
device ID, not a patient ID, so the data has to be linked through an enrollment log, and the linker audits
that join and reports any device it cannot match. Lows are rare, about 4% of prediction windows, and
metformin-only patients almost never go low, which is what you would expect clinically.

The link between record and sensor is therefore synthetic, by design, and we say so in the code, in
every document and on the dashboard banner. There is a subtle consequence: our simulator makes the
record drive glucose. A higher HbA1c gives bigger spikes, and sulfonylureas and weak kidneys give more
lows. So any benefit we measure from the record on this data is partly built in. Keep that in mind,
because it comes back later in the video.

## 6. Built to be trusted (7:30-8:30) · slide 6

**Narration**

Five habits make our numbers trustworthy.

One: we split by patient. No patient appears in both training and test, and the code raises an error
if one does. Splitting by time instead leaks a person's own patterns into the test.

Two: we score onset windows, meaning predictions made before glucose crosses the line. Predicting a low
once the patient is already low is trivial and inflates every metric.

Three: every comparison comes with a 95% confidence interval from a patient-level bootstrap, and model
comparisons are paired.

Four: every choice, such as alert thresholds and the alert policy, was made on validation patients. The
test patients were touched once.

Five: everything is reproducible. Fixed seeds, one command rebuilds the data, the models and every
report, and 128 automated tests check it.

## 7. Results on synthetic patients (8:30-10:10) · slide 7

**Narration**

On 60 synthetic test patients, both events are predictable. For lows we report AUPRC, because lows are
rare and plain accuracy would mislead. Chance is about 0.02. Extrapolating the trend scores 0.19,
LightGBM on CGM 0.65, and with the record 0.71. For highs: 0.38, 0.72 and 0.74. AUROC is 0.93 for highs
and 0.99 for lows.

Two results we did not expect. The deep GRU fusion model did not beat LightGBM, and steps, heart rate
and sleep added nothing measurable on top of glucose. So we deploy LightGBM: it is the best for lows,
it is fast, it explains itself exactly, and its probabilities are well calibrated.

And a warning we give ourselves: these synthetic numbers are optimistic, because simulated lows drift in
slowly and are easy to see coming.

We also audited the model by subgroup: sex, age, kidney function, drug class, HbA1c and BMI. For highs,
AUROC stays between 0.92 and 0.95 in every group. For lows, one group stands out. Insulin users' lows are
the hardest to predict, with an AUPRC of 0.22 against 0.72 for sulfonylurea users, although that group is
small. And one group cannot be audited at all: only one test patient has reduced kidney function, even
though weak kidneys raise the risk of lows. We report both, because an audit that only shows good news is
not an audit.

## 8. Alerts a clinic can live with (10:10-12:10) · slide 8

**Narration**

Accuracy is not enough. Alarm fatigue is what kills alerting systems in hospitals. If we raised an alert
every time the risk crossed a threshold, a patient would generate about three alerting windows a day,
and only 18% of low alerts would be real.

So we built an alert policy: one alert when risk first crosses the threshold, then silence for 90
minutes while the risk stays high, and never an alert while the CGM already reads low, because the
sensor's own alarm covers that. The thresholds were chosen on validation patients.

On the test patients, the policy warns of 76% of low-glucose episodes at least 30 minutes ahead, with a
median warning of 65 minutes. That is about one low alert every two days per patient, and two out of
three alerts are followed by a real low. The trade-off is explicit: scoring every window would catch 98%,
at six times the alerts. Where to sit on that curve is a clinical decision, and our tooling shows the
whole curve.

At night, 71% of lows are warned at least 30 minutes ahead, with about one alert every eleven nights.
For highs, 68% of episodes are warned 30 minutes ahead, at about two and a half alerts a day. We also
tested more elaborate policies, such as re-arming only after the risk falls. On validation patients it
made things worse, because a slowly drifting low keeps the risk high and the alert never re-arms, so we
kept the simple policy.

## 9. Live dashboard demo (12:10-17:20) · switch to the browser

**On screen:** the dashboard, patient P0014, "Highest hypoglycaemia risk". If the live demo fails while
recording, show slides 9-11 instead: they hold stills of the same three moments (replay, drivers, what-if).

**Narration and actions**

*(Hero card and risk cards visible.)* This is the clinician dashboard. At the top is the patient: a
66-year-old woman on metformin and a sulfonylurea, HbA1c 7.1%. The two cards show her risks for the next
two hours. Hypoglycaemia risk is 95%, well above the threshold, and the alert policy has raised an alert.
Hyperglycaemia risk is zero.

*(Live replay tab, press Play.)* The first tab is live replay. It plays her morning forward as if it were
happening now, and the model only ever sees data up to the moving line. Watch the glucose fall after
eight o'clock. *(Wait for 08:25.)* At 08:25 the hypo risk crosses the threshold and the alert appears,
with its two main reasons: her sulfonylurea dose, and glucose falling about one milligram per decilitre
per minute. *(Let it run to about 09:45.)* Her glucose goes below 70 at 09:40, 75 minutes after the
alert. *(Scroll down.)* Below the chart, the dashboard lists every alert in this window with its lead
time, and it would list a missed episode or a false alert in exactly the same way.

*(Glucose timeline tab.)* The timeline shows the bigger picture: glucose with the high and low bands,
both risks over time, every alert, and the next two hours after "now", drawn dotted because the model
has not seen them.

*(Alert drivers tab.)* Here is why the model raised this risk. Each bar is that input's exact
contribution: current glucose, the sulfonylurea dose, time of day, the lowest reading in the last hour.
They are also grouped by data stream. These explain the model, not her physiology, and the page says so.

*(What-if tab, press "Start the what-if twin", choose "Extra insulin dose".)* Now the twin. It was fitted
to her own first week of data. Four physiology parameters were estimated from her glucose and her logged
meals and doses. I ask what happens with an extra 8-unit insulin dose an hour ago. The risk model's
answer is on the cards, and the simulated glucose is in the chart: her projected low drops from about
64 to 55. The shaded area is the 80% uncertainty band. On patients the band never saw, the real CGM
fell inside it 79% of the time. *(Point at the caption.)* The band covers the projection as logged; how
the twin responds to a change adds error the band does not include, and the page says that too.

*(Switch the scenario to "Skipped meal".)* A skipped meal works the same way: the twin removes the meal
from her day and re-simulates, and the risk model is re-run on what it would have seen. *(Switch "Twin
physiology" to Oracle for a moment.)* For comparison we keep an oracle mode that replays the simulator's
hidden truth. No real system could do this, and the page says so. It shows how close the personalised twin
gets. *(Switch back to Personalised.)*

*(Sidebar: "Choose a time", pick an evening, e.g. 20:00.)* A clinician can pick any moment in the record.
In the evening, the "Late dinner" preset moves her dinner two hours later with her medicine at the usual
time, which is a very common real-life pattern.

*(Patient overview tab.)* Finally, the overview: her record, latest labs, and 14-day CGM metrics against
the international consensus targets.

## 10. Then we tested it on real people (17:20-19:00) · slides 12-13

**Narration**

Synthetic results are only as good as the simulator, so we tested on real data. Two public datasets,
never used for training.

The first is CGMacros from PhysioNet: 14 adults with Type 2 diabetes in the US. We applied our model
unchanged. It still ranks risk better than chance and beats trend extrapolation, but AUROC for highs
drops from 0.93 on synthetic patients to 0.70 on real ones. Real glucose is noisier and less regular than
our simulator's. We show that gap rather than hide it.

The probabilities were also off: the model underpredicted. Recalibrating on the other real
participants, without retraining, cuts the calibration error from 0.136 to 0.024. That is the first step
of any real deployment, and we have shown it works.

We then asked a follow-up question: when only a handful of real patients are available, is our synthetic
data still worth having? We trained on 2, 4, 7 or 10 real participants, with and without the synthetic
data, as pretraining or mixed in. The answer is: not reliably. With two real participants, mixing in
synthetic data helps a little. From four upwards it adds nothing, and its one consistent benefit is
steadier results between random draws. Real data overtakes the simulator after four to seven participants.
That tells us where the value is: the simulator is for building and testing the system, and real data is
for the final model.

## 11. Real linked records, and what the EHR is really for (19:00-21:50) · slides 14-16

**Narration**

The second dataset is the one that tests our central idea. ShanghaiT2DM, published in Scientific Data
in 2023, has 100 real patients with Type 2 diabetes, and for each one the CGM comes with their own
clinical record and a log of their insulin and tablets. Real linked data, the thing we could not get
for India.

We trained on it with patient-grouped cross-validation, so every patient is scored by a model that never
saw them. The CGM model predicts real lows well: AUROC 0.89, and 0.91 at night, with good calibration.
And our synthetic-trained model, applied unchanged, beats trend extrapolation on these real lows, by 0.07
AUPRC. So the simulator taught something real.

Then the key question: does adding the record help? We tried our 24 EHR fields, then the logged doses,
then the full clinical record with C-peptide and other labs. None of them improved the forecast. We even
tried a simple model that cannot memorise patients, in case the trees were overfitting. Still nothing.

So we asked where the record should matter most: a new patient, with little or no CGM yet. With no CGM at
all, the record does carry a real signal. AUROC goes from chance to 0.54, which is enough to decide who
to watch, but far too weak to raise alerts on. With just 30 minutes of CGM, the CGM model already reaches
0.88, almost as good as six hours, and from then on the record adds nothing.

So the record is a prior, not a forecaster. It helps decide who needs a sensor, it personalises the
twin, and it gives the clinician context. It does not improve a two-hour forecast once glucose is being
measured.

*(Slide 16.)* This changed our project. We had a result on synthetic data saying the record improves the
forecast. Real data said otherwise, so we changed the claim in our README, our model card and this
video. What we claim now is on the left. What we no longer claim is on the right. We think this is the
most important slide in the deck, because it is what a clinical team needs to see before trusting any
model.

## 12. Path to real use in India (21:50-23:40) · slides 17-18

**Narration**

Nothing here is ready for patients, and we have written down what would make it ready. There are four
evidence steps. First, a retrospective study on real linked records from Indian clinics, with ethics
approval under ICMR's guidelines for AI in healthcare, which must test the record's value directly rather
than assume it. Then recalibration to the local population. Then a silent pilot, where the model scores in
the background and outcomes are compared afterwards. Finally a pilot with diabetologists in the loop.

Around that is the compliance frame: consent and security under the DPDP Act and its 2025 Rules, ABDM and
ABHA consent flows to link a patient's record to their device, and CDSCO's rules for medical device
software. The CGM's own alarms always stay on. The twin supplements them and never replaces them.

Once running, it would need the operations any clinical model needs: monitoring of data quality and sensor
changes, checks that the scores and their calibration do not drift, versioned models with the evidence for
each version recorded, a route for reporting missed lows and harmful alerts, and a subgroup audit at every
release, adding region, diet and device type, which our synthetic data does not have.

*(Slide 18.)* And it is built to be checked, not just demoed: 128 automated tests, one command to rebuild
everything, two public real datasets checked, a live dashboard, an MIT licence, and a model card in the
repository.

## 13. Limitations and close (23:40-24:40) · slides 19-20

**Narration**

We know the limits. The deployed model has never been trained on real patients. Our real checks are small,
14 US and 100 Chinese patients, not Indian outpatients. Insulin users' lows are the hardest to predict, on
synthetic and real data alike. And a real twin needs meals and doses logged when they happen, because the
model cannot see a planned dinner or a fresh dose until glucose moves.

*(Slide 20.)* To close: a twin that warns early, explains itself, and is honest about what it knows. Our next
step is validation on real linked records from Indian clinics, with clinicians in the loop. If you can
help with access to such data or a diabetologist partner, we would love to talk.

Thank you.
