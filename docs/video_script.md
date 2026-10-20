# Video script (about 25 minutes, one presenter)

*Glycaemic Digital Twin: Happiest Health Digital Twin Challenge 2026*

The challenge needs a video of at least 20 minutes (a PowerPoint recording or an unlisted YouTube link).
This script is for **one presenter** and follows the order of the final video:

1. **Cold open** (16 s): the live-replay clip `video_assets/live_replay_cold_open.mp4`, with your voice over it
2. **Slides 1-8** recorded in PowerPoint: problem, solution, how it works, results, alerting
3. **Live demo** (4:48): the recorded clip `video_assets/live_demo.mp4`, with your voice over it
4. **Slides 12-20** recorded in PowerPoint: real patients, what we claim, path forward, close

Slides 9-11 (replay, drivers, what-if stills) are not used: the live demo covers them. Keep them as a backup.
Every slide's narration below is also in that slide's **speaker notes** in `docs/presentation.pptx`, so it shows
in PowerPoint's Record view while you read.

**How to read the script.** *Do* lines are actions. **Say** lines are spoken. Speak at a calm pace (about 130
words a minute); the times assume that pace. Every number here matches the slides, the README and the reports in
`docs/`. Say "synthetic" whenever a result comes from synthetic patients, and never imply clinical validation.

---

## Part 1 · Cold open (0:00-0:16) · clip `live_replay_cold_open.mp4`

*Do:* record this voice separately and lay it over the clip. The alert appears at **3.7 s** and the low at **10.5 s**.

**Say**

(0.0 s) Watch this patient's morning, replayed as it happened.
(3.7 s, as the alert appears) At 8:25, our digital twin warned that her glucose was heading dangerously low.
(10.5 s, as the low begins) At 9:40, it did. Seventy-five minutes of warning.
(13.5 s) Here's how I built it, and how I tested it on real patients.

---

## Part 2 · Slides 1-8 (0:16-11:06) · PowerPoint recording

### Slide 1 · Title (0:16-1:14)

*Do:* camera on. Look at the lens, not the notes, for the first sentence.

**Say**

Hello, I'm *[your name]*, from team *[team name]* at *[college or incubator]*. My project for the Happiest Health
Digital Twin Challenge is a glycaemic digital twin for people with Type 2 diabetes.

In one sentence: it's a computer copy of a patient that combines their health record with their wearable data,
warns of dangerous high and low glucose two hours before it happens, explains every warning, and lets a doctor
test "what if" on the twin before trying it on the patient.

And then I did the part I think matters most: I checked every claim on real patient data, and where real data
disagreed with me, I changed the claim. Everything you'll see is a research prototype, not for clinical use.

### Slide 2 · The problem (1:14-2:41)

**Say**

India has about 101 million people with diabetes, according to the ICMR-INDIAB study, most of them with Type 2.

Day to day, glucose control fails in two opposite directions. The first is the spike after a meal. Traditional
Indian meals are rich in carbohydrates: rice, roti, idli, poha. With insulin resistance they become large, long
spikes above 180 milligrams per decilitre, and those repeated spikes drive the long-term complications.

The second is the low, below 70. Sulfonylureas are widely prescribed in India because they're cheap and they
work, and together with insulin they can push glucose too low, after a skipped meal, extra activity, or an extra
dose. The worst case is at night, when the patient is asleep and can't react.

The doctor sees an HbA1c every three months. These events happen within hours. The information to see them
coming exists, but in two places that are never combined: the health record, with diagnoses, labs and medicines,
and the wearable sensor, with a glucose reading every five minutes. So nobody turns them into an early warning a
doctor can act on. That's the gap.

### Slide 3 · Our answer (2:41-3:57)

**Say**

My answer is a digital twin of each patient, with four capabilities.

Predict: every 30 minutes, from the last six hours of wearable data plus the record, it estimates the risk of
glucose going above 180 or below 70 in the next two hours.

Alert: a risk score alone would fire constantly, so an alert policy raises one alert per risk episode.

Explain: every risk comes with its main reasons, like "glucose is falling" or "this patient takes a sulfonylurea".

Simulate: the twin is fitted to the patient's own data, so a clinician can ask what a late dinner, a skipped
meal or an extra dose would do, and see the answer with an honest uncertainty band.

That last part is what makes it a twin rather than just a predictor. A predictor tells you what will probably
happen. A twin is a copy of this particular patient's body that you can experiment on, safely, before changing
anything for the real person.

### Slide 4 · Architecture (3:57-5:18)

**Say**

Here's how it fits together, in five layers from left to right.

Layer one is the data. A synthetic health-record cohort of 300 patients drives a physiology simulator, which
produces the wearable stream: CGM every five minutes with realistic sensor lag, noise and dropouts, plus steps,
heart rate and sleep. An enrollment log links each patient to a device, the way a real study would.

Layer two links the streams and builds features: six-hour windows of sensor data, and the record as a static
vector. Patients are split into training, validation and test sets as whole people, never by time.

Layer three is the models: a simple trend baseline, LightGBM, and a deep fusion network for comparison.

Layer four is evaluation. The teal box is the important one: real external data, used to check the models and
to train a real-patient model.

Layer five is the twin itself: the patient state, the explanations, the personalised what-if twin, a dashboard
for clinicians, and a FHIR API so other hospital systems can use it.

### Slide 5 · Data (5:18-6:40)

**Say**

I need to be upfront about the data. No public dataset links hospital records and wearable data for the same
Indian patients. So I built a physiology simulator: a Bergman-style glucose model with Indian meals, drug
pharmacokinetics for metformin, sulfonylureas and insulin, sleep, activity and the morning rise in glucose. It
generates 300 patients, 14 days each.

A few details make it more than a toy. Doctors titrate doses, so patients with worse control take stronger drugs,
as in real clinics. The CGM isn't the true glucose: it lags by about ten minutes, it's noisy and it drops out,
like a real sensor. And lows are rare, about 4% of prediction windows, which is realistic.

The link between record and sensor is therefore synthetic by design, and I say so in the code, the documents and
on the dashboard. There's a subtle consequence: my simulator makes the record drive glucose. So any benefit I
measure from the record on this data is partly built in. Keep that in mind, because it comes back later.

### Slide 6 · Rigour (6:40-7:29)

**Say**

Five habits make these numbers trustworthy.

One: I split by patient. No patient is in both training and test, and the code raises an error if one is.

Two: I score onset windows, predictions made before glucose crosses the line. Predicting a low once the patient
is already low is trivial, and it inflates every metric.

Three: every comparison has a 95% confidence interval from a patient-level bootstrap.

Four: every choice, like the alert thresholds, was made on validation patients. The test patients were touched
once.

Five: it's reproducible. Fixed seeds, one command rebuilds everything, and 140 automated tests run on every push.

### Slide 7 · Results on synthetic patients (7:29-8:58)

**Say**

On 60 synthetic test patients, both events are predictable. A quick word on the metrics. AUROC asks how well the
model separates "trouble coming" from "fine": 0.5 is a coin flip and 1.0 is perfect. AUPRC asks how many of its
warnings are right while still catching the events, which is the fairer measure when events are rare. For lows I
report AUPRC, because lows are rare and accuracy would mislead: chance is about 0.02. Extrapolating the trend scores 0.19. LightGBM on CGM scores 0.65,
and with wearables and the record, 0.71. For highs: 0.38, 0.72 and 0.74. AUROC is 0.93 for highs and 0.99 for lows.

Two results I didn't expect. A deep GRU fusion network didn't beat LightGBM, and steps, heart rate and sleep added
nothing measurable on top of glucose. So I deploy LightGBM: it's the best for lows, it's fast, it explains itself
exactly, and its probabilities are well calibrated.

And a warning I give myself: these synthetic numbers are optimistic, because simulated lows drift in slowly and
are easy to see coming. I measured exactly how slowly, and I'll show you that in the real-data part.

### Slide 8 · Alerting, synthetic and real (8:58-11:06)

**Say**

Accuracy isn't enough. Alarm fatigue is what kills alerting systems in hospitals. So the alert policy fires once
when risk crosses the threshold, then stays quiet for 90 minutes while the risk stays high, and never fires while
the CGM already reads low, because the sensor's own alarm covers that.

The teal bars are 60 synthetic test patients. The policy warns of 76% of lows at least 30 minutes ahead, with a
median warning of 65 minutes, about one alert every two days, and two out of three alerts followed by a real low.

The coral bars are what matters more: the same policy on 100 real patients from ShanghaiT2DM, with a model
trained on real data and thresholds set on other patients. It warns of 35% of real lows at least 30 minutes ahead,
and 53% at 15 minutes, at about 0.6 alerts a day, with 44% of alerts followed by a real low.

The policy is a trade-off, and the tooling shows the whole curve. Raising an alert in every high-risk window
would catch 98% of synthetic lows 30 minutes ahead, but at about three alerting windows a day, and only about one
in five would be real. One alert per episode gives six times fewer alerts for some loss of sensitivity. Where to sit
on that curve is a clinical decision, not mine.

As for the real-patient numbers: that's a big drop, and I'm showing it on purpose. Real lows arrive faster than simulated ones, so the synthetic
figure is an upper bound, and the real one is the honest baseline to improve. Now let me show you the product.

---

## Part 3 · Live demo (11:06-15:54) · clip `live_demo.mp4`

*Do:* lay your voice over the clip. Times are **clip time** (0:00 = start of the demo clip); start each block when
its moment appears. The clip uses synthetic patient P0014.

**(0:00-0:24) Hero and risk cards**
This is the clinician dashboard. At the top is the patient: a 66-year-old woman on metformin and a sulfonylurea,
HbA1c 7.1%. The two cards show her risk for the next two hours. Hypoglycaemia risk is 95%, well above the
threshold, so an alert is raised. Hyperglycaemia risk is zero.

**(0:24-0:46) Live replay, press Play**
The first tab is live replay. It plays her morning forward as if it were happening now, and the model only ever
sees data up to the moving line. I'll press play. Watch the glucose start to fall after eight o'clock.

**(0:46-1:01) Paused at the 08:25 alert**
I'll pause here. At 8:25 the risk crosses the threshold and the alert appears, with its two reasons: her
sulfonylurea dose, and glucose falling almost one milligram per decilitre a minute.

**(1:01-1:14) Paused at 09:40**
And at 9:40 her glucose goes below 70. That's 75 minutes after the alert: time to eat something, or to call her.

**(1:14-1:29) Replay summary**
Below the chart, the dashboard lists every alert in this window with its lead time. A missed low or a false
alert would be listed the same way.

**(1:29-1:55) Glucose timeline**
The timeline tab shows the bigger picture: glucose with the high and low bands, both risks over time, and every
alert. After "now", the next two hours are drawn dotted, because the model hasn't seen them. This is what a
clinician would scan in a review.

**(1:55-2:27) Alert drivers**
Here's why the model raised this risk. Each bar is that input's exact contribution: current glucose at 70, the
sulfonylurea dose, the time of day, and the lowest reading in the last hour. The contributions are also grouped
by data stream. They explain the model, not her physiology, and the page says so.

**(2:27-2:39) What-if: start the twin**
Now the twin. It was fitted to her own first week of data: four physiology parameters, from her glucose and
her logged meals and doses.

**(2:39-3:12) Extra insulin dose, band and caption**
What if she took an extra 8 units of insulin an hour ago? The risk model's answer is on the cards, and the
twin's projection is in the chart: her low drops from about 64 to 55. The shaded area is the 80% uncertainty
band: on patients it never saw, glucose fell inside it 79% of the time. The caption states its limit: it
doesn't cover the extra error of the change itself.

**(3:12-3:28) Skipped meal**
A skipped meal works the same way. The twin removes the meal from her day, re-simulates, and the risk model is
re-run on what it would have seen.

**(3:28-3:40) Oracle mode**
For comparison, an oracle mode replays the simulator's hidden truth. No real system could do this, and the
page says so. It shows how close the twin gets.

**(3:40-3:51) Back to personalised, poor sleep**
Back to the personalised twin. Or a poor night's sleep, which raises insulin resistance the next day.

**(3:51-4:28) Choose a time, late dinner**
A clinician can pick any moment in the record. Here's an evening earlier in her fortnight. The late-dinner preset
moves her dinner two hours later while her medicine stays at the usual time, a very common pattern in real life.
The cards show what the model would predict, and the chart shows what the twin expects her glucose to do. Notice
the warning: the model can't see a meal that hasn't happened yet, and the dashboard says so.

**(4:28-4:48) Patient overview**
Finally, the overview: her record, her latest labs, and her 14-day CGM metrics. Seventeen percent of her time is
below 70, against a consensus target under four percent. That's exactly the patient this twin is built for.

---

## Part 4 · Slides 12-20 (15:54-24:55) · PowerPoint recording

### Slide 12 · Part two: real people (15:54-16:27)

**Say**

That's the product. But synthetic results are only as good as the simulator, so I tested on real people: two
public, de-identified datasets. Neither is ever committed to the repository: they're downloaded separately, under their own licences. CGMacros, with 14 US adults with Type 2 diabetes. And ShanghaiT2DM, with 100 patients whose CGM
comes with their own clinical records, which is exactly what I couldn't get for India.

### Slide 13 · Real CGM (16:27-17:52)

**Say**

First, CGMacros. I applied my synthetic-trained model unchanged. It still ranks risk better than chance and beats
trend extrapolation, but AUROC for highs drops from 0.93 on synthetic patients to 0.70 on real ones. Retrained on
the real participants, it reaches 0.77.

I didn't stop at the gap; I measured where it comes from. Comparing the simulator with both real cohorts, the
level of glucose, time in range and the rate of highs and lows are realistic. But real lows fall about twice as
fast, roughly half a milligram per decilitre per minute against a quarter in the simulator, real swings are
bigger, and real patients peak after breakfast. That's a concrete to-do list for the simulator.

The probabilities were also off: the model underpredicted. So I recalibrated it the way a hospital would. For
each participant, a small correction curve is fitted on all the other participants and applied to that one, so no
one is corrected using their own data. Without any retraining, that cuts the calibration error from 0.136 to 0.024.
That's step one of any real deployment.

### Slide 14 · Real linked records (17:52-18:59)

**Say**

ShanghaiT2DM tests the central idea. A hundred real patients, with CGM, their clinical records, and a log of
their insulin and tablets.

I trained on it with patient-grouped cross-validation: the patients are split into ten groups, and each group is
scored by a model trained on the other nine, so every patient is scored by a model that never saw them.
The CGM model predicts real lows well: AUROC 0.89, and 0.91 at night, with good calibration. My synthetic-trained
model, applied unchanged, still beats trend extrapolation on these real lows, so the simulator taught something
real.

Then the key question: does adding the record help? I tried the 24 record fields, then the logged doses, then the
full clinical record with C-peptide and other labs. None of them improved the forecast. I even tried a simple model
that can't memorise patients. Still nothing.

### Slide 15 · What the record is for (18:59-20:14)

**Say**

So I asked where the record should matter most: a new patient, with little or no CGM yet. To test it, I took the
same windows and hid all but the last few hours of glucose: none at all, 30 minutes, one hour, two hours, or the full
six. Then I compared CGM alone with CGM plus the record at each step.

With no CGM at all, the record does carry a real signal: AUROC goes from chance to 0.54. That's enough to decide
who to watch, but far too weak to raise alerts on. With just 30 minutes of CGM, the CGM model already reaches
0.88, almost as good as six hours, and from then on the record adds nothing.

So the record is a prior, not a forecaster. It helps decide who needs a sensor, it personalises the twin, and it
gives the clinician context. It doesn't improve a two-hour forecast once glucose is being measured.

### Slide 16 · What I claim, and no longer claim (20:14-21:20)

**Say**

This changed my project. On synthetic data, adding the record improved the forecast. Real data said otherwise,
so I changed the claim, in the README, the model card and this video.

On the left is what I claim: real lows predicted two hours ahead on 100 real patients, an honest real-world
alerting baseline, recalibration that works, a twin with honest uncertainty, and a product that ships a model
trained on real patients.

On the right is what I no longer claim. That adding the record improves the two-hour forecast: that was true on
synthetic data only. That the synthetic accuracy of 0.93 is what real patients would see. And any clinical validity
at all: this is a research prototype.

I think this is the most important slide, because it's what a clinical team needs to see before trusting any model.

### Slide 17 · Path to real use in India (21:20-22:28)

**Say**

Nothing here is ready for patients, and I've written down what would make it ready. Four steps: a study on real
linked records from Indian clinics, with ethics approval under ICMR's AI guidelines, which must test the record's
value rather than assume it. Then recalibration to the local population. Then a silent pilot, where the model scores
in the background and nobody acts on it, so we learn how it behaves on live data without any risk to patients.
Then a pilot with diabetologists in the loop, judged on usefulness, workload and safety.

Around that is the compliance frame: consent and security under the DPDP Act and its 2025 Rules, ABDM and ABHA
consent flows to link a record to a device, and CDSCO's rules for medical device software. And the CGM's own
alarms always stay on: the twin supplements them and never replaces them.

### Slide 18 · Engineering (22:28-23:28)

**Say**

It's built to be checked, not just demoed: 140 automated tests that run on every push, one command to rebuild the
data, models and reports, two public real datasets checked, and a live dashboard.

It can also plug into hospital systems. A FHIR API takes a patient's record and CGM readings in FHIR, the standard
behind India's ABDM health records, and returns standard risk assessments with their reasons. I tested that a
patient sent through the API gets exactly the dashboard's risk.

And it's open: an MIT licence, a model card that states what the model is for and what it isn't, a written
deployment path, and every result table in the repository. The real datasets stay out of it, downloaded separately
under their own licences.

### Slide 19 · Limitations (23:28-24:17)

**Say**

I know the limits, and each one points to the next piece of work. The default model is trained on synthetic
data, and the real-trained one has seen only 100 patients: the fix is more real data. Those patients are Chinese and
American, not Indian outpatients: the fix is the Indian clinic study. Insulin users' lows are the hardest to predict,
on synthetic and real data alike: that needs more insulin users and better dose logging. And a real twin needs meals
and doses logged as they happen, because the model can't see a planned dinner or a fresh dose until glucose moves.

### Slide 20 · Close (24:17-24:55)

*Do:* camera on.

**Say**

To close. I set out to build a digital twin that warns early, explains itself, and is honest about what it
knows. It predicts real lows, it explains every alert, it lets a clinician test a change safely, and it says
clearly where the evidence stops. The next step is
validation on real linked records from Indian clinics, with clinicians in the loop. If you can help with that
data, or a diabetologist partner, I'd love to talk. Thank you.
