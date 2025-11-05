
# ✅ Temporal Models — Q1–Q7 with Solutions

---

## **Q1. One-step transition**

Given the Markov chain:

* (P(Sun \mid Sun) = 0.8)
* (P(Rain \mid Sun) = 0.2)

If the initial distribution is (P(X_1) = [1, 0]) (100% Sun), compute (P(X_2)).

### ✅ **Solution**

[
P(X_2 = Sun) = 1 \cdot 0.8 = 0.8
]
[
P(X_2 = Rain) = 1 \cdot 0.2 = 0.2
]

**Answer:**
[
P(X_2) = (0.8,; 0.2)
]

---

## **Q2. Stationary Distribution**

Given transitions:

* (P(Sun|Sun)=0.8)
* (P(Rain|Sun)=0.2)
* (P(Sun|Rain)=0.4)
* (P(Rain|Rain)=0.6)

Find the stationary distribution (\pi = (\pi_s, \pi_r)) such that (\pi = \pi T).

### ✅ **Solution**

Solve:
[
\pi_s = 0.8 \pi_s + 0.4 \pi_r
]
[
\pi_r = 0.2 \pi_s + 0.6 \pi_r
]

From the first:
[
\pi_s - 0.8\pi_s = 0.4\pi_r
]
[
0.2\pi_s = 0.4\pi_r
]
[
\pi_s = 2\pi_r
]

Normalization:
[
\pi_s + \pi_r = 1
]
[
2\pi_r + \pi_r = 1
]
[
3\pi_r = 1 \Rightarrow \pi_r = \tfrac{1}{3}
]
[
\pi_s = \tfrac{2}{3}
]

**Answer:**
[
\pi = \big(\tfrac{2}{3},; \tfrac{1}{3}\big)
]

---

## **Q3. Markov Order**

Explain the difference between a **first-order** and **second-order** Markov process.

### ✅ **Solution**

* **First-order:**
  [
  P(X_t \mid X_{0:t-1}) = P(X_t \mid X_{t-1})
  ]
  Depends only on previous **1** state.

* **Second-order:**
  [
  P(X_t \mid X_{0:t-1}) = P(X_t \mid X_{t-2}, X_{t-1})
  ]
  Depends on the previous **2** states.

---

## **Q4. Stationarity Assumption**

What does the **stationarity assumption** in Markov models mean?

### ✅ **Solution**

Stationarity means the **transition** and **sensor** probabilities do **not change over time**.
The model uses the **same CPTs for all time steps**.

---

## **Q5. Two-step transition**

Given initial (P(X_1 = Sun)=0.5), transitions:

* (P(Rain|Sun)=0.2)
* (P(Rain|Rain)=0.6)

Compute (P(X_2 = Rain)).

### ✅ **Solution**

[
P(X_2 = Rain)
= 0.5 \cdot 0.2 + 0.5 \cdot 0.6
]
[
= 0.1 + 0.3 = 0.4
]

**Answer:**
[
P(X_2 = Rain) = 0.4
]

---

## **Q6. Why initial influence disappears**

Explain why the influence of the initial state diminishes over time.

### ✅ **Solution**

Because repeated application of transitions “mixes” probability mass.
Under a regular Markov chain, the distribution converges to a **stationary distribution**, losing memory of the starting state.
Only the transition matrix determines the long-run behavior.

---

## **Q7. Transition Diagram**

Draw the transition diagram for the chain:

* (P(Sick|Healthy)=0.1)
* (P(Healthy|Sick)=0.4)

### ✅ **Solution**

Diagram (text form):

```
Healthy  --0.1-->  Sick
   ^                 |
   |                 v
   <------0.4------- Healthy
```

Self-loops (complements):

* (P(Healthy|Healthy)=0.9)
* (P(Sick|Sick)=0.6)

---

