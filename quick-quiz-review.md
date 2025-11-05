
---

# 🧭 Topic Map

| Exam Topic                                        | Covered in                        | Core Ideas                                                       |
| ------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| **Learning Agent**                                | ch 18 (§1–3)                      | performance element, learning element, critic, problem generator |
| **Ockham’s Razor**                                | ch 18 p 12 & slides 15 – 16       | simplicity vs consistency                                        |
| **Decision Trees & Pruning**                      | ch 18 p 22 – 31 + part2 (Pruning) | entropy, information gain, Chi-square, early stopping            |
| **Overfitting / Validation**                      | part2 (Overfitting)               | training vs test error, cross-validation                         |
| **Parametric vs Non-parametric (LDA, KDE / GMM)** | not in these, in other slides     | —                                                                |
| **HMM / DBN**                                     | Temporal Prob Models              | already done earlier                                             |

---

Below are **10 quiz-style Q&A blocks per topic**, formatted GitHub-friendly.

---

# 🧩 Topic 1 – Learning Agent Model

```
1. What are the four components of a learning agent?
→ Performance element, Learning element, Critic, Problem generator.

2. Role of performance element?
→ Chooses actions based on current percepts and internal state.

3. Role of learning element?
→ Improves the performance element using feedback from the critic.

4. Role of critic?
→ Provides performance feedback by comparing behavior to a standard.

5. Role of problem generator?
→ Proposes exploratory actions to discover new, informative experiences.

6. Difference between supervised and reinforcement feedback?
→ Supervised gives correct labels each instance; RL gives occasional rewards.

7. How does learning modify the agent?
→ Adjusts decision mechanisms to improve future performance.

8. Why learning useful for system design?
→ Lets environment teach the agent instead of hard-coding rules.

9. What dictates the design of the learning element?
→ Type of performance element, component to learn, representation, feedback type.

10. Example mapping:
→ Utility-based agent learns percept→action function; feedback = reward signal.
```

---

# 🪶 Topic 2 – Ockham’s Razor and Inductive Learning

```
1. Quote meaning:
   “Entities should not be multiplied unnecessarily” → prefer simpler models.

2. Why simplicity helps?
   → Reduces risk of overfitting; better generalization.

3. In inductive learning, what is f(x)?
   → Target function mapping inputs to outputs.

4. What is hypothesis h?
   → Learner’s approximation of f using training data.

5. When is h consistent?
   → When it agrees with all training examples.

6. Relation of Ockham’s Razor to decision trees?
   → Prefer smaller trees consistent with data.

7. What is bias–variance intuition here?
   → Simpler models have higher bias, lower variance; complex opposite.

8. Give an example of inductive bias.
   → “The simplest consistent hypothesis is best.”

9. What assumptions real inductive learning simplifies?
   → Deterministic f, observable inputs, given examples, agent wants to learn f.

10. Why curve-fitting demonstrates Ockham’s Razor?
   → A smoother curve explaining data points is preferred to jagged overfit.
```

---

# 🌳 Topic 3 – Decision Tree Learning & Information Gain

```
1. Aim:
   → Find the smallest tree consistent with training examples.

2. Recursion base cases:
   • All examples same class → return class.
   • No examples → return default.
   • No attributes → return mode class.

3. Attribute selection criterion?
   → Information Gain = H(parent) − Remainder(A).

4. Entropy formula:
   H(p, n) = −p/(p+n) log₂(p/(p+n)) − n/(p+n) log₂(n/(p+n))

5. Remainder(A):
   Σ_i (p_i + n_i)/(p+n) × H(p_i, n_i)

6. Choose attribute with smallest Remainder → highest InfoGain.

7. Example: Patrons? vs Type? (Restaurant)
   Patrons? Gain ≈ 0.54 bits > Type? 0 bits → choose Patrons?.

8. What happens with many attributes?
   → Larger hypothesis space → risk of overfitting.

9. Decision tree expressiveness?
   → Can represent any Boolean function (2⁽²ⁿ⁾ possible trees).

10. Why prefer compact tree?
   → Simpler ⇒ better generalization (Ockham’s Razor).
```

---

# ✂️ Topic 4 – Decision Tree Pruning (Chi-Square / Validation)

```
1. Purpose of pruning?
   → Reduce overfitting by removing statistically insignificant splits.

2. Two strategies:
   → Post-pruning (generate then prune) vs Early stopping (halt before split).

3. Chi-square test null hypothesis?
   → “No real pattern” between attribute and class.

4. Compute χ²:
   χ² = Σ_k ( (observed_k − expected_k)² / expected_k )

5. Degrees of freedom (df):
   df = (#positive + #negative classes − 1) × (#attribute values − 1)

6. Example threshold:
   → df=3 → critical 7.82 at 5%; if χ² > 7.82, keep split; else prune.

7. Why not just remove low-gain attributes directly?
   → Combination of low-gain attributes may still classify jointly (e.g., XOR).

8. Effect of noise:
   → Pruning helps by ignoring spurious correlations.

9. What is Early stopping criterion?
   → Stop if split doesn’t improve InfoGain significantly on validation set.

10. Trade-off:
   → Too much pruning = underfit; too little = overfit.
```

---

# 📈 Topic 5 – Overfitting & Model Generalization / Validation

```
1. Define overfitting:
   → Model fits training noise; performs poorly on unseen data.

2. Define underfitting:
   → Model too simple to capture data patterns.

3. Typical symptom:
   → Training accuracy ↑, test accuracy ↓ as epochs continue.

4. How to visualize?
   → Plot train/test loss vs epochs; test loss bottoms then rises.

5. Cross-validation (k-fold):
   → Split dataset into k parts; train k times each leaving one fold out.

6. Purpose of validation set:
   → Tune hyperparameters and detect overfitting before test phase.

7. How does hypothesis space size affect overfitting?
   → Larger space → higher variance → greater risk.

8. Give numeric example:
   → Acc(train)=99%, Acc(test)=60% → classic overfit.

9. How to combat overfitting?
   → Simplify model, regularization, pruning, or add data.

10. Why Ockham’s Razor applies again?
   → Simpler model less likely to memorize noise ⇒ better generalization.
```

---

# 🧠 Topic 6 – Temporal Models (HMM / DBN Recap)

(Already derived from previous “Section B”)

```
1. Markov assumption → P(X_t | X_0:t−1) = P(X_t | X_{t−1})
2. Sensor Markov → P(E_t | X_0:t, E_0:t−1) = P(E_t | X_t)
3. Filtering → update belief with evidence e_t
4. Prediction → project belief forward without new evidence
5. Smoothing → re-estimate past given future data
6. Forward algorithm → Elapse + Observe + Normalize
7. Stationary distribution → π = πT
8. HMM components → (π₁, A, B)
9. Viterbi = most likely state sequence
10. DBN = multi-variable generalization of HMM.
```

---

