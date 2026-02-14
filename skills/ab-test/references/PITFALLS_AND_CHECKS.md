# Pitfalls and Validity Checks

Before accepting any A/B test result, run through this list of "Intuition Busters" to ensure validity.

## 1. Sample Ratio Mismatch (SRM)

**Definition:** When the observed allocation of users differs statistically from the designed allocation (e.g., 50.5% vs 49.5% in a large sample).

* **Why it happens:** Usually a bug. Example: The "Treatment" variant crashes on old Android phones. Those users never send the "assignment event" and are excluded from the Treatment pool.
* **Result:** The Treatment pool is now "cleaner" (better devices) than Control, creating **Survivorship Bias**. Treatment wins due to population quality, not feature quality.


* **Action:** Discard the experiment. Do not try to "fix" the data.

## 2. The Novelty & Primacy Effects

**Definition:** Temporary changes in user behavior caused by the "newness" of the change, not its utility.

* **Novelty Effect:** Clicks go up initially because the button looks different. Fades over time.
* **Primacy Effect:** Metrics dip initially because users are fighting muscle memory. Recovers over time.
* **Detection:** Plot the treatment effect () over time. If the lift is consistently decaying, it is likely a novelty effect.

## 3. Peeking (Continuous Monitoring)

**Definition:** Checking significance daily and stopping the moment .

* **Risk:** Increases False Positive Rate (Type I Error) from 5% to ~19% or higher.


* **Solution:** Use **Sequential Testing** (e.g., O'Brien-Fleming boundaries) or Bayesian methods if you need to stop early. Otherwise, stick to the fixed duration calculated in the design phase.

## 4. Twyman's Law

**Definition:** *"Any figure that looks interesting or different is usually wrong."* 

* **Heuristic:** If a test shows a +20% lift in a mature metric (like Revenue), it is almost certainly a data error.
* **Checklist for "Amazing" Results:**
1. Is there an SRM?
2. Are bots/crawlers skewing the data?
3. Is logging duplicated in the Treatment?
4. Did we break the "Unsubscribe" flow (increasing retention artificially)?