


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

