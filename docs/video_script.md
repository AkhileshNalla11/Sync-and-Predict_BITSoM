# Video script: "The twin that was proven wrong, and got better"

*Glycaemic Digital Twin: Happiest Health Digital Twin Challenge 2026. One presenter, about 24 minutes (about 26 at a slow pace).*

The video tells one true story in three acts. **Act 1, the build:** the twin works on synthetic patients.
**Act 2, the test:** real patients prove three of its claims partly wrong. **Act 3, the rebuild:** what changed, and
what we can honestly claim now. Every number matches the slides, the README and the reports in `docs/`.

You record yourself speaking each **take** below. The video is then assembled around your takes: the animated
opening, chapter cards, slides, the recorded demo, animated charts, claim stamps and captions are all added in the
edit, so you only need to talk to the camera.

---

## How to record your takes

**Camera.** Landscape, 1080p, 30 fps (a phone on a stand at eye level, or a good webcam). Sit centred, framed from
the chest up, with a little space above your head. In the edit you appear in a card beside the slides, cropped
to a portrait shape, so stay near the centre of the frame and don't lean out of it.

**Light and sound.** Face a window or a lamp; no bright window behind you. A plain, tidy background. Use a quiet
room with soft furnishings and a mic close to you (wired earphones with a mic, or a clip-on mic). No fan or AC
hum. Silence your phone.

**Each take is its own recording.** Press record, wait two seconds, speak, wait two seconds, stop. If you stumble,
stop and record the whole take again: they're short. Keep only the good version of each take.

**File names.** Name each file after its take, for example `V04.mp4` or `D07.mp4`, and put them all in one folder,
`recordings/`, in the project (it's ignored by git). If renaming is a pain, record the takes in the order listed
and keep the camera's file names: the order is enough.

**Delivery.** Talk to the lens as if to one judge across a table. Calm, about 130 words a minute. Smile on the
first and last takes. You don't need to memorise anything: read from the script on a screen just below the
camera, and look up at the start of each sentence. Wear the same clothes for every take.

**Voice-only takes.** `V01` and all the demo takes (`D01`-`D15`) are used for their sound only: still record
them on camera (it keeps the sound identical to the other takes), but you can read them straight from the screen.

**Say the words as written.** The on-screen captions and the timing of the animations (the stamps, the cards, the
charts) are made from this script. Small slips are fine, but don't rephrase sentences or add new ones: the captions
would then show different words from the ones you say.

**Saying the names.** ShanghaiT2DM: "Shanghai T-two-D-M". CGMacros: "C-G-Macros". CGM: "C-G-M". AUROC: "A-U-R-O-C"
(or "area under the curve"). AUPRC: "A-U-P-R-C". HbA1c: "H-B-A-one-C". FHIR: "fire". eGFR: "e-G-F-R". DPDP, ABDM,
ABHA, CDSCO, ICMR: letter by letter. BITSoM: "bits-om". Decimals as digits: 0.89 is "point eight nine".

**Before the full build.** Once the takes are in `recordings/`, a quick check runs first (about a minute). It lists
any missing take, a take recorded in portrait, sound that is too quiet or clipped, and a take whose length doesn't
fit its script (a wrong or cut-off take), so you can redo just those.

### Take list

| Take | Part | On screen in the video | About |
|---|---|---|---|
| V01 | Opening | Animated replay of one patient's morning (your voice only) | 0:20 |
| V02 | Opening | You, full screen, with the key numbers appearing beside you | 1:45 |
| V03-V08 | Act 1 | Slides 2-7, with you in a card beside them | 7:15 |
| V09 | Act 1 | Animated alerting chart, synthetic patients | 1:15 |
| D01-D15 | Act 1 | The recorded dashboard demo (your voice only) | 4:46 |
| V10 | Act 1 | Three claims appear, waiting to be tested | 0:30 |
| V11-V15 | Act 2 | Slide 12, animated real-vs-synthetic chart, a real missed low, slides 13 and 14, with stamps | 4:50 |
| V16-V21 | Act 3 | Slide 15, the claims with stamps, slides 17-19, then you full screen to close | 5:55 |

---

## Opening

### V01 · Cold open (voice only)

Pause for a full second between the lines: the animation is timed to them.

> Watch this patient's morning, replayed as it happened.
>
> At 8:25, our digital twin warned that her glucose was heading dangerously low.
>
> At 9:40, it did. Seventy-five minutes of warning.
>
> Here's how I built it, how I tested it on real patients, and what real data proved wrong.

### V02 · Hello, and the whole story in 90 seconds (you, full screen)

> Hello, I'm Akhilesh Nalla, from team Sync and Predict at BITSoM. I built a glycaemic digital twin for
> people with Type 2 diabetes. If you only have ninety seconds, here's the whole story.
>
> The twin combines a patient's health record with their wearable data. Every thirty minutes it estimates the risk
> of a dangerous high or low in the next two hours. It raises one alert per risk episode, explains why, and lets a
> doctor try a change, like a late dinner or an extra dose, on the twin before trying it on the patient.
>
> On synthetic patients, it warned of three out of four lows at least half an hour ahead.
>
> Then I tested it on a hundred real patients. Real lows arrive faster, and it warned of about one in three.
> And adding the health record, which I expected to help, didn't improve the forecast at all.
>
> So I changed my claims. I shipped a model trained on real patients, and I found what the record is really for: a
> new patient with no glucose data yet.
>
> That's the story. I built it, real data proved me partly wrong, and it got better. The chapters are in the
> description, so you can jump to any part. Everything you'll see is a research prototype, not for clinical use.
> Here's the full version.

---

## Act 1 · The build

### V03 · Slide 2, the problem

> India has about 101 million people with diabetes, according to the ICMR-INDIAB study, most of them with Type 2.
>
> Day to day, glucose control fails in two opposite directions. The first is the spike after a meal. Traditional
> Indian meals are rich in carbohydrates: rice, roti, idli, poha. With insulin resistance they become large, long
> spikes above 180 milligrams per decilitre, and those repeated spikes drive the long-term complications.
>
> The second is the low, below 70. Sulfonylureas are widely prescribed in India because they're cheap and they
> work, and together with insulin they can push glucose too low, after a skipped meal, extra activity, or an extra
> dose. The worst case is at night, when the patient is asleep and can't react.
>
> The doctor sees an HbA1c every three months. These events happen within hours. The information to see them
> coming exists, but in two places that are never combined: the health record, with diagnoses, labs and medicines,
> and the wearable sensor, with a glucose reading every five minutes. So nobody turns them into an early warning a
> doctor can act on. That's the gap.

### V04 · Slide 3, the answer

> My answer is a digital twin of each patient, with four capabilities.
>
> Predict: every 30 minutes, from the last six hours of wearable data plus the record, it estimates the risk of
> glucose going above 180 or below 70 in the next two hours.
>
> Alert: a risk score alone would fire constantly, so an alert policy raises one alert per risk episode.
>
> Explain: every risk comes with its main reasons, like "glucose is falling" or "this patient takes a sulfonylurea".
>
> Simulate: the twin is fitted to the patient's own data, so a clinician can ask what a late dinner, a skipped
> meal or an extra dose would do, and see the answer with an honest uncertainty band.
>
> That last part is what makes it a twin rather than just a predictor. A predictor tells you what will probably
> happen. A twin is a copy of this particular patient's body that you can experiment on, safely, before changing
> anything for the real person.

### V05 · Slide 4, architecture

> Here's how it fits together, in five layers from left to right.
>
> Layer one is the data. A synthetic health-record cohort of 300 patients drives a physiology simulator, which
> produces the wearable stream: CGM every five minutes with realistic sensor lag, noise and dropouts, plus steps,
> heart rate and sleep. An enrollment log links each patient to a device, the way a real study would.
>
> Layer two links the streams and builds features: six-hour windows of sensor data, and the record as a static
> vector. Patients are split into training, validation and test sets as whole people, never by time.
>
> Layer three is the models: a simple trend baseline, LightGBM, and a deep fusion network for comparison.
>
> Layer four is evaluation. The teal box is the important one: real external data, used to check the models and
> to train a real-patient model.
>
> Layer five is the twin itself: the patient state, the explanations, the personalised what-if twin, a dashboard
> for clinicians, and a FHIR API so other hospital systems can use it.

### V06 · Slide 5, data

> I need to be upfront about the data. No public dataset links hospital records and wearable data for the same
> Indian patients. So I built a physiology simulator: a Bergman-style glucose model with Indian meals, drug
> pharmacokinetics for metformin, sulfonylureas and insulin, sleep, activity and the morning rise in glucose. It
> generates 300 patients, 14 days each.
>
> A few details make it more than a toy. Doctors titrate doses, so patients with worse control take stronger drugs,
> as in real clinics. The CGM isn't the true glucose: it lags by about ten minutes, it's noisy and it drops out,
> like a real sensor. And lows are rare, about 4% of prediction windows, which is realistic.
>
> The link between record and sensor is therefore synthetic by design, and I say so in the code, the documents and
> on the dashboard. There's a subtle consequence: my simulator makes the record drive glucose. So any benefit I
> measure from the record on this data is partly built in. Keep that in mind, because it comes back later.

### V07 · Slide 6, rigour

> Five habits make these numbers trustworthy.
>
> One: I split by patient. No patient is in both training and test, and the code raises an error if one is.
>
> Two: I score onset windows, predictions made before glucose crosses the line. Predicting a low once the patient
> is already low is trivial, and it inflates every metric.
>
> Three: every comparison has a 95% confidence interval from a patient-level bootstrap.
>
> Four: every choice, like the alert thresholds, was made on validation patients. The test patients were touched
> once.
>
> Five: it's reproducible. Fixed seeds, one command rebuilds everything, and 140 automated tests run on every push.

### V08 · Slide 7, results on synthetic patients

> On 60 synthetic test patients, both events are predictable. A quick word on the metrics. AUROC asks how well the
> model separates "trouble coming" from "fine": 0.5 is a coin flip and 1.0 is perfect. AUPRC asks how many of its
> warnings are right while still catching the events, which is the fairer measure when events are rare.
>
> For lows, chance is about 0.02. Extrapolating the trend scores 0.19. LightGBM on CGM scores 0.65, and with
> wearables and the record, 0.71. For highs: 0.38, 0.72 and 0.74. AUROC is 0.93 for highs and 0.99 for lows.
>
> Two results I didn't expect. A deep GRU fusion network didn't beat LightGBM, and steps, heart rate and sleep added
> nothing measurable on top of glucose. So I deploy LightGBM: it's the best for lows, it's fast, it explains itself
> exactly, and its probabilities are well calibrated.

### V09 · Alerting on synthetic patients (animated chart)

> Accuracy isn't enough. Alarm fatigue is what kills alerting systems in hospitals. So the alert policy fires once
> when risk crosses the threshold, then stays quiet for 90 minutes while the risk stays high, and never fires while
> the CGM already reads low, because the sensor's own alarm covers that.
>
> On 60 synthetic test patients, the policy warns of 76% of lows at least 30 minutes ahead, with a median warning of
> 65 minutes. That's about one alert every two days, and two out of three alerts are followed by a real low.
>
> It's a trade-off, and the tooling shows the whole curve. Raising an alert in every high-risk window would catch
> 98% of lows, but at about three alerting windows a day, and only about half would be real. One alert per
> episode gives six times fewer alerts, and two out of three are real. Where to sit on that curve is a clinical
> decision, not mine.
>
> Now let me show you the product.

### D01-D15 · The live demo (voice only)

Record each block as its own take. In the video, each block starts at the moment shown in brackets (clip time in
`video_assets/live_demo.mp4`); if a block runs a little long, the edit holds the picture until you finish. The
patient is synthetic patient P0014.

**D01 (0:00) Hero and risk cards**
> This is the clinician dashboard. At the top is the patient: a 66-year-old woman on metformin and a sulfonylurea,
> HbA1c 7.1%. The two cards show her risk for the next two hours. Hypoglycaemia risk is 95%, well above the
> threshold, so an alert is raised. Hyperglycaemia risk is zero.

**D02 (0:24) Live replay, press Play**
> The first tab is live replay. It plays her morning forward as if it were happening now, and the model only ever
> sees data up to the moving line. I'll press play. Watch the glucose start to fall after eight o'clock.

**D03 (0:45) Paused at the 08:25 alert**
> I'll pause here. At 8:25 the risk crosses the threshold and the alert appears, with its two reasons: her
> sulfonylurea dose, and glucose falling almost one milligram per decilitre a minute.

**D04 (1:01) Paused at 09:40**
> And at 9:40 her glucose goes below 70. That's 75 minutes after the alert: time to eat something, or to call her.

**D05 (1:13) Replay summary**
> Below the chart, the dashboard lists every alert in this window with its lead time. A missed low or a false
> alert would be listed the same way.

**D06 (1:28) Glucose timeline**
> The timeline tab shows the bigger picture: glucose with the high and low bands, both risks over time, and every
> alert. After "now", the next two hours are drawn dotted, because the model hasn't seen them. This is what a
> clinician would scan in a review.

**D07 (1:54) Alert drivers**
> Here's why the model raised this risk. Each bar is that input's exact contribution: current glucose at 70, the
> sulfonylurea dose, the time of day, and the lowest reading in the last hour. The contributions are also grouped
> by data stream. They explain the model, not her physiology, and the page says so.

**D08 (2:25) What-if: start the twin**
> Now the twin. It was fitted to her own first week of data: four physiology parameters, from her glucose and
> her logged meals and doses.

**D09 (2:36) Extra insulin dose, band and caption**
> What if she took an extra 8 units of insulin an hour ago? The risk model's answer is on the cards, and the
> twin's projection is in the chart: her low drops from about 64 to 55. The shaded area is the 80% uncertainty
> band: on patients it never saw, glucose fell inside it 79% of the time. The caption states its limit: it
> doesn't cover the extra error of the change itself.

**D10 (3:09) Skipped meal**
> A skipped meal works the same way. The twin removes the meal from her day, re-simulates, and the risk model is
> re-run on what it would have seen.

**D11 (3:25) Oracle mode**
> For comparison, an oracle mode replays the simulator's hidden truth. No real system could do this, and the
> page says so. It shows how close the twin gets.

**D12 (3:37) Back to personalised, poor sleep**
> Back to the personalised twin. Or a poor night's sleep, which raises insulin resistance the next day.

**D13 (3:48) Choose a time, late dinner**
> A clinician can pick any moment in the record. Here's an evening earlier in her fortnight. The late-dinner preset
> moves her dinner two hours later while her medicine stays at the usual time, a very common pattern in real life.
> The cards show what the model would predict, and the chart shows what the twin expects her glucose to do.

**D14 (4:12) The meal it can't see**
> Notice the warning: the model can't see a meal that hasn't happened yet, and the dashboard says so.

**D15 (4:26) Patient overview**
> Finally, the overview: her record, her latest labs, and her 14-day CGM metrics. Seventeen percent of her time is
> below 70, against a consensus target under four percent. That's exactly the patient this twin is built for.

### V10 · Three claims to test

> So that's the twin, on synthetic patients. It seems to support three claims. One: it warns of most lows in time.
> Two: adding the health record makes the forecast better. Three: my simulator behaves like real patients.
>
> Every one of them comes from synthetic data. So I tested all three on real people.

---

## Act 2 · The test

### V11 · Slide 12, real people

> Synthetic results are only as good as the simulator, so I used two public, de-identified datasets. Neither is
> ever committed to the repository: they're downloaded separately, under their own licences. CGMacros, with 14 US
> adults with Type 2 diabetes. And ShanghaiT2DM, with 100 patients whose CGM comes with their own clinical records,
> which is exactly what I couldn't get for India.

### V12 · Claim one: does it warn in time? (animated chart)

> Claim one. The same alert policy, on 100 real ShanghaiT2DM patients, with a model trained on real data and
> thresholds set on other patients.
>
> It warns of 35% of real lows at least 30 minutes ahead, and 53% at 15 minutes. The median warning falls from 65
> minutes to 25. It raises about 0.6 alerts a day, and 44% of them are followed by a real low.
>
> That's a big drop, and I'm showing it on purpose. Claim one: revised. The synthetic figure is an upper bound, and
> the real one is the honest baseline to improve.

### V13 · Why: a real low the twin saw too late (animated traces)

> Here's why, in one picture. On the left is the simulated low from the start of this video. The twin raised its
> alert at 8:25, seventy-five minutes ahead, while glucose was still 138.
>
> On the right is a real low from ShanghaiT2DM. This patient's glucose peaked at 214 after breakfast, then fell by
> more than 130 in an hour and a half. At the 9:47 check, the risk was rising, but still under the threshold. At the
> next check, half an hour later, the patient was already low. No alert came in time.
>
> Across all the real lows, glucose falls almost twice as fast in the hour before, about half a milligram per
> decilitre per minute, against a quarter in my simulator. Faster lows leave less warning. And this case points to
> the first fix I'd test: the model checks every 30 minutes, and a fall this fast may need a check every five.

### V14 · Claim three: is the simulator realistic? (slide 13)

> Claim three, the simulator. First, CGMacros. I applied my synthetic-trained model unchanged. It still ranks risk
> better than chance and beats trend extrapolation, but AUROC for highs drops from 0.93 on synthetic patients to
> 0.70 on real ones. Retrained on the real participants, it reaches 0.77.
>
> Then I compared the simulator with both real cohorts, metric by metric. The level of glucose, time in range and
> the rate of highs and lows are realistic. But real lows fall faster, as you just saw, real swings are bigger, and
> real patients peak after breakfast. Claim three: revised. And that comparison is now a to-do list for the
> simulator.
>
> The probabilities were also off: the model underpredicted. So I recalibrated it the way a hospital would. For
> each participant, a small correction curve is fitted on all the other participants and applied to that one, so no
> one is corrected using their own data. Without any retraining, that cuts the calibration error from 0.136 to
> 0.024. That's step one of any real deployment.

### V15 · Claim two: does the record help? (slide 14)

> Claim two is the central idea, and ShanghaiT2DM tests it: a hundred real patients, with CGM, their clinical
> records, and a log of their insulin and tablets.
>
> I trained on it with patient-grouped cross-validation: the patients are split into ten groups, and each group is
> scored by a model trained on the other nine, so every patient is scored by a model that never saw them. The CGM
> model predicts real lows well: AUROC 0.89, and 0.91 at night, with good calibration. My synthetic-trained model,
> applied unchanged, still beats trend extrapolation on these real lows, so the simulator taught something real.
>
> Then the key question: does adding the record help? I tried the 24 record fields, then the logged doses, then the
> full clinical record with C-peptide and other labs. None of them improved the forecast. I even tried a simple
> model that can't memorise patients. Still nothing. Claim two: dropped.

---

## Act 3 · The rebuild

### V16 · Slide 15, what the record is for

> So I asked where the record should matter most: a new patient, with little or no CGM yet. To test it, I took the
> same windows and hid all but the last few hours of glucose: none at all, 30 minutes, one hour, two hours, or the
> full six. Then I compared CGM alone with CGM plus the record at each step.
>
> With no CGM at all, the record does carry a real signal: AUROC rises from 0.41, no better than chance, to 0.54.
> That's enough to decide who to watch, but far too weak to raise alerts on. With just 30 minutes of CGM, the CGM model already reaches
> 0.88, almost as good as six hours, and from then on the record adds nothing.
>
> So the record is a prior, not a forecaster. It helps decide who needs a sensor, it personalises the twin, and it
> gives the clinician context. It doesn't improve a two-hour forecast once glucose is being measured.

### V17 · What I claim now (claims with stamps)

> This changed my project. Real data disagreed with me, so I changed the claims, in the README, the model card and
> this video.
>
> Here's what I claim now. A twin built on CGM predicts real lows two hours ahead, on 100 real patients. Its alerts
> give an honest real-world baseline. Recalibration on local data fixes the probabilities. The personalised twin
> comes with honest uncertainty. And the product ships a model trained on real patients: in the dashboard, a
> clinician can switch to it with one click.
>
> And here's what I no longer claim. That adding the record improves the two-hour forecast: that was true on
> synthetic data only. That the synthetic accuracy of 0.93 is what real patients would see. And any clinical validity
> at all: this is a research prototype.
>
> I think this is the most important minute of the video, because it's what a clinical team needs to see before
> trusting any model.

### V18 · Slide 17, path to real use in India

> Nothing here is ready for patients, and I've written down what would make it ready. Four steps: a study on real
> linked records from Indian clinics, with ethics approval under ICMR's AI guidelines, which must test the record's
> value rather than assume it. Then recalibration to the local population. Then a silent pilot, where the model
> scores in the background and nobody acts on it, so we learn how it behaves on live data without any risk to
> patients. Then a pilot with diabetologists in the loop, judged on usefulness, workload and safety.
>
> Around that is the compliance frame: consent and security under the DPDP Act and its 2025 Rules, ABDM and ABHA
> consent flows to link a record to a device, and CDSCO's rules for medical device software. And the CGM's own
> alarms always stay on: the twin supplements them and never replaces them.

### V19 · Slide 18, engineering

> It's built to be checked, not just demoed: 140 automated tests that run on every push, one command to rebuild the
> data, models and reports, two public real datasets checked, and a live dashboard.
>
> It can also plug into hospital systems. A FHIR API takes a patient's record and CGM readings in FHIR, the standard
> behind India's ABDM health records, and returns standard risk assessments with their reasons. I tested that a
> patient sent through the API gets exactly the dashboard's risk.
>
> And it's open: an MIT licence, a model card that states what the model is for and what it isn't, a written
> deployment path, and every result table in the repository.

### V20 · Slide 19, limitations

> I know the limits, and each one points to the next piece of work. The default model is trained on synthetic
> data, and the real-trained one has seen only 100 patients: the fix is more real data. Those patients are Chinese
> and American, not Indian outpatients: the fix is the Indian clinic study. Insulin users' lows are the hardest to
> predict, on synthetic and real data alike: that needs more insulin users and better dose logging. And a real twin
> needs meals and doses logged as they happen, because the model can't see a planned dinner or a fresh dose until
> glucose moves.

### V21 · Close (you, full screen)

> To close. I set out to build a digital twin that warns early, explains itself, and is honest about what it
> knows. I built it, real data proved me partly wrong, and it got better. It predicts real lows, it explains every
> alert, it lets a clinician test a change safely, and it says clearly where the evidence stops.
>
> The next step is validation on real linked records from Indian clinics, with clinicians in the loop. If you can
> help with that data, or a diabetologist partner, I'd love to talk. Thank you.

---

## Backup

If the demo clip can't be used, slides 9-11 (replay, drivers, what-if stills) cover the same ground. Slides 1, 8
and 16 are replaced in the video by the animated opening, the alerting charts and the claim stamps, and stay in the
deck for anyone reading it.
