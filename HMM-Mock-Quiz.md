


# Temporal Probability Models — Q1–Q7 (with Answers)

> Scope: Markov chains, stationary distribution, order, stationarity, short numeric transitions, intuition, tiny diagram.

---

## Q1 — One-step transition

**Given:**

* P(Sun | Sun) = 0.8
* P(Rain | Sun) = 0.2
* Initial P(X₁) = [1, 0] (100% Sun)

**Task:** Compute P(X₂).

**Answer:**

```
P(X₂ = Sun)  = 1 * 0.8 = 0.8
P(X₂ = Rain) = 1 * 0.2 = 0.2

P(X₂) = (0.8, 0.2)
```

---

## Q2 — Stationary distribution

**Transitions:**

* P(Sun|Sun)=0.8, P(Rain|Sun)=0.2
* P(Sun|Rain)=0.4, P(Rain|Rain)=0.6

**Task:** Find π = (π_s, π_r) such that π = π · T and π_s + π_r = 1.

**Answer (derivation):**

```
π_s = 0.8 π_s + 0.4 π_r
π_s - 0.8 π_s = 0.4 π_r
0.2 π_s = 0.4 π_r  ⇒  π_s = 2 π_r

π_s + π_r = 1  ⇒  2π_r + π_r = 1  ⇒  3π_r = 1  ⇒  π_r = 1/3
π_s = 2/3

π = (2/3, 1/3)
```

**Quick checks:**

```
Flow Sun→Rain:  (2/3)*0.2 = 2/15
Flow Rain→Sun:  (1/3)*0.4 = 2/15   (balanced)

π·T:
Sun' = (2/3)*0.8 + (1/3)*0.4 = 2/3
Rain'= (2/3)*0.2 + (1/3)*0.6 = 1/3  ✔
```

---

## Q3 — Markov order

**Task:** Explain first- vs second-order Markov.

**Answer:**

```
First-order:  P(X_t | X_0: t-1) = P(X_t | X_{t-1})      (depends on 1 prev state)
Second-order: P(X_t | X_0: t-1) = P(X_t | X_{t-2}, X_{t-1}) (depends on 2 prev states)
```

---

## Q4 — Stationarity assumption

**Task:** What does “stationarity” mean here?

**Answer:**

```
Transition P(X_t | X_{t-1}) and sensor P(E_t | X_t) are time-invariant (same CPTs for all t).
(Not about t→∞; it’s about parameters not changing across time steps.)
```

---

## Q5 — Two-step transition probability

**Given:** initial P(X₁=Sun)=0.5, transitions:

* P(Rain|Sun)=0.2
* P(Rain|Rain)=0.6

**Task:** Compute P(X₂=Rain).

**Answer:**

```
P(X₂=Rain) = P(Sun)*P(Rain|Sun) + P(Rain)*P(Rain|Rain)
            = 0.5*0.2 + 0.5*0.6
            = 0.1 + 0.3
            = 0.4
```

---

## Q6 — Why initial influence fades

**Task:** Explain the intuition.

**Answer:**

```
Repeated application of the transition matrix “mixes” the distribution.
For regular chains, P(X_t) → stationary π regardless of the start.
Memory of the initial state decays; long-run behavior is governed by T.
```

---

## Q7 — Transition diagram (text)

**Given:**

* P(Sick|Healthy)=0.1
* P(Healthy|Sick)=0.4
  (So P(Healthy|Healthy)=0.9, P(Sick|Sick)=0.6)

**Answer (ASCII):**

```
Healthy --0.1--> Sick
   ^               |
   |               v
   <--0.4------- Healthy
```

---

Here is the **Section B (HMM + DBN)** question-and-answer set, formatted cleanly for **GitHub README**.
Readable, no LaTeX reliance, only Markdown + code blocks + simple HTML.

---

# ✅ Section B — HMMs & DBNs (Q8–Q25)

---

## **Q8 — HMM filtering (unnormalized first step)**

**Given:**
States: Rain, Sun
Evidence: `U` (Umbrella)
Transitions:

* P(Rain|Rain)=0.7, P(Sun|Rain)=0.3
* P(Rain|Sun)=0.2, P(Sun|Sun)=0.8
  Emissions:
* P(U|Rain)=0.9
* P(U|Sun)=0.2
  Initial: P(Rain)=0.5, P(Sun)=0.5

**Task:** Compute unnormalized P(X₁ | e₁ = U).

**Answer:**

```
P(Rain, U) = 0.5 * 0.9 = 0.45
P(Sun,  U) = 0.5 * 0.2 = 0.10

Unnormalized posterior = (0.45, 0.10)
```

---

## **Q9 — Sensor Markov assumption**

**Answer:**

```
P(E_t | X_0:t, E_0:t-1) = P(E_t | X_t)

Current evidence depends ONLY on the current hidden state, nothing else.
```

---

## **Q10 — Hidden vs observed variables in HMM**

**Answer:**

```
Hidden (X_t): unobserved true state (e.g., weather, robot position).
Observed (E_t): noisy evidence emitted by the state (e.g., umbrella, sensor reading).
```

---

## **Q11 — Prediction step**

**Given belief:**
B(Xₜ) = [Sun=0.6, Rain=0.4]

**Transition matrix T:**

```
[ 0.7  0.3 ]   # Sun->(Sun,Rain)
[ 0.4  0.6 ]   # Rain->(Sun,Rain)
```

**Task:** Predict B(Xₜ₊₁) before evidence.

**Answer:**

```
P(Sun')  = 0.6*0.7 + 0.4*0.4 = 0.42 + 0.16 = 0.58
P(Rain') = 0.6*0.3 + 0.4*0.6 = 0.18 + 0.24 = 0.42

Predicted = (0.58, 0.42)
```

---

## **Q12 — Why normalization matters**

**Answer:**

```
After multiplying by evidence likelihoods, values are NOT probabilities anymore.
Without normalization, P(X_t | e_1:t) won't sum to 1.
Belief state becomes invalid; future updates produce wrong results.
```

---

## **Q13 — Filtering recursion**

**Answer (standard forward formula):**

```
P(X_t | e_1:t) ∝ P(e_t | X_t) * Σ_{x_{t-1}} [ P(X_t | X_{t-1}) * P(X_{t-1} | e_1:t-1) ]
```

---

## **Q14 — Two-step filtering structure**

**Answer:**

```
1. Elapse time:
   Prediction:   P(X_t) = Σ_x P(X_t | X_{t-1}) * P(X_{t-1}|e_1:t-1)

2. Observation:
   Correction:   Multiply by P(e_t | X_t) and normalize.
```

---

## **Q15 — Define filtering / prediction / smoothing**

**Answer:**

```
Filtering:  P(X_t | e_1:t)
            Uses all evidence up to t.

Prediction: P(X_{t+k} | e_1:t)
            No evidence after t.

Smoothing:  P(X_k | e_1:t)  for k < t
            Uses future evidence as well.
```

---

## **Q16 — Most likely explanation**

**Task:** For evidence sequence e₁:₃ = {U, U, ¬U}, what algorithm finds the single best hidden-state path?

**Answer:**

```
The Viterbi algorithm.
Returns argmax over the entire sequence X_1:3.
```

---

## **Q17 — When HMM reduces to Markov chain**

**Answer:**

```
When observations are perfectly informative (deterministic), or when you ignore them entirely.
Then hidden states become directly observed → simple Markov chain.
```

---

## **Q18 — Correcting misconception**

**Claim:** “E_t depends on E_{t-1}.”

**Answer:**

```
Wrong. Evidence nodes have no edges between them.
Correct: E_t depends ONLY on X_t.
```

---

## **Q19 — Robot tracking state & observation**

**Answer:**

```
State X_t      = robot's true position on grid (hidden)
Observation E_t = sensor readings (range, lidar, distance)
```

---

## **Q20 — Unnormalized posterior update**

**Given:**
P(Xₜ | e₁:ₜ) = [0.3, 0.7]
Next evidence favors state1 by factor 2: L = [2, 1]

**Answer:**

```
Unnormalized = [0.3*2, 0.7*1] = (0.6, 0.7)
```

---

## **Q21 — How DBN generalizes HMM**

**Answer:**

```
HMM = 1 hidden variable X_t and 1 evidence E_t per slice.
DBN = multiple variables per time slice, arbitrary dependencies.

Example extra variables: robot orientation, velocity, multiple sensors.
```

---

## **Q22 — Computational issue with unrolling DBNs**

**Answer:**

```
Exact inference cost grows with number of time steps t.
Graph becomes huge; variable elimination becomes intractable.
```

---

## **Q23 — Roll-up filtering**

**Answer:**

```
Add slice t+1, incorporate transitions, then sum-out (eliminate) slice t.
Keeps constant-size belief representation over time.
```

---

## **Q24 — Why likelihood weighting fails in long DBNs**

**Answer:**

```
Weight of samples that match evidence decays exponentially with time.
Most samples become effectively weight = 0 → degeneracy.
```

---

## **Q25 — Four inference tasks with examples**

**Answer:**

```
1. Filtering     → tracking a speaker's mouth position live.
2. Prediction    → forecasting tomorrow's weather.
3. Smoothing     → correcting yesterday’s robot position using later data.
4. MLE / Viterbi → finding spoken word sequence in speech recognition.
```

---


