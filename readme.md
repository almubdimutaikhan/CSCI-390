# 📘 Stationary Distribution (Steady-State Probabilities)

### 🔹 Definition
The **stationary distribution** is a probability vector **π = [P<sub>∞</sub>(A), P<sub>∞</sub>(B), P<sub>∞</sub>(C)]**  
such that the system stays unchanged after transitions:

**π = π T**

---

### 🔹 Core Equation
For every state *i*:
> P(i) = Σ P(j) · P(j → i)

and  
> P(A) + P(B) + P(C) = 1  

---

### 🔹 Steps to Find π

1. **Write transition matrix T** (rows = from-state, columns = to-state).  
2. **Form balance equations**  
   - P(A) = P(A) P(A→A) + P(B) P(B→A) + P(C) P(C→A)  
   - P(B) = P(A) P(A→B) + P(B) P(B→B) + P(C) P(C→B)  
   - P(C) = P(A) P(A→C) + P(B) P(B→C) + P(C) P(C→C)
3. **Add normalization:** P(A)+P(B)+P(C)=1  
4. **Solve** the linear system for P(A), P(B), P(C).

---

### 🔹 Notes
- Represents the **long-run fraction of time** spent in each state.  
- Exists if the chain is **ergodic** (connected + aperiodic).  
- Equivalent to the **eigenvector of Tᵀ** with eigenvalue 1.  

---

### 🧠 Example
If P(A)=P(B) and P(C)=0.5 P(A):  
→ 2 P(A)+0.5 P(A)=1 → P(A)=0.4  
✅ P∞(A)=0.4, P∞(B)=0.4, P∞(C)=0.2



## 🌳 Decision Tree — Information Gain (Quick Revision)

### 1️⃣ Step 1 — Compute Overall Entropy

Target variable (class) = Z

$$
H(Z) = -\sum_i p_i \log_2 p_i
$$

* Measures **uncertainty** in Z.
* Example: 5 True, 5 False → (H(Z)=1).

---

### 2️⃣ Step 2 — Compute Conditional Entropy for each Attribute

For each attribute (A):

* Split the data by A’s values (e.g., A=0 and A=1).
* For each subset, compute entropy of Z inside it.

$$
H(Z|A) = \sum_{v \in Values(A)} P(A=v) , H(Z|A=v)
$$

where
$$
H(Z|A=v) = -\sum_i P(Z=i | A=v) \log_2 P(Z=i | A=v)
$$

✅ Weighted average:
Each subset entropy is weighted by its proportion in total data.

---

### 3️⃣ Step 3 — Compute Information Gain

$$
IG(Z, A) = H(Z) - H(Z|A)
$$

* The **higher the IG**, the more that attribute reduces uncertainty.
* Choose the **attribute with the largest IG** as the root.

---

### 4️⃣ Step 4 — Recursive Split

* Repeat steps 1–3 **within each branch**, using remaining attributes.
* Stop when:

  * All examples in a node have the same class, or
  * No attributes left (or IG = 0).

---

### 🧠 Quick Intuition

| Term                 | Meaning                                               |
| -------------------- | ----------------------------------------------------- |
| **Entropy**          | How mixed the classes are                             |
| **Information Gain** | How much splitting reduces uncertainty                |
| **High IG**          | Attribute gives strong signal about target            |
| **Low IG (≈0)**      | Attribute doesn’t help — same mixture after splitting |

---

### ⚡ Formula Summary

$$
\begin{aligned}
H(Z) &= -\sum p_i \log_2 p_i \
H(Z|A) &= \sum_v P(A=v) H(Z|A=v) \
IG(Z,A) &= H(Z) - H(Z|A)
\end{aligned}
$$

---

### ✅ Tip for Fast Exam Solving

1. Count positives and negatives for Z.
2. Compute base entropy.
3. Split by each attribute.
4. Compute branch entropies → weighted sum.
5. Pick the one with **max IG**.

---



# 🤖 AI Quiz 2 — Summary Notes

## 1️⃣ Bayesian Networks & Exact Inference

**Core ideas**

* Conditional independence; chain rule factorization of joint $P(X_1,\dots,X_n)=\prod_i P(X_i|\text{Parents}(X_i))$.
* **Query = wanted vars; Evidence = known values; Hidden = others → eliminate.**
* **Variable elimination**:

  1. Fix evidence.
  2. Multiply factors that share a var.
  3. Sum out that var.
  4. Repeat for hidden vars only.
* **Order choice:** use *min-fill* or *min-degree* to keep factors small.
* **Enumeration vs Elimination:** both exact; elimination is faster.
* **Sampling:**

  * *Prior* – top-down sampling (no evidence).
  * *Rejection* – discard inconsistent samples.
  * *Likelihood Weighting* – keep all, weight by evidence likelihood.

---

## 2️⃣ Temporal / Hidden Markov Models

* **Markov assumption:** $P(X_t|X_{t-1},…,X_0)=P(X_t|X_{t-1})$.
* **Filtering:** alternate

  * *Elapse time* – predict next state.
  * *Observe evidence* – update with $P(E_t|X_t)$.
* **Forward algorithm:** recursively apply Elapse→Observe to maintain belief state.
* **Stationary distribution:** solve $P_\infty=P_\infty T$ (sum = 1).

---

## 3️⃣ Decision Trees / Statistical Learning

* **Entropy:** $H(S)=-\sum p_i\log_2p_i$.
* **Conditional entropy:** $H(S|A)=\sum_v P(A=v)H(S|A=v)$.
* **Information Gain:** $IG(S,A)=H(S)-H(S|A)$.
* **Build tree:** choose attr. with max IG → recurse.
* **Stop:** pure node or IG = 0.
* **Overfitting:** deeper = lower bias / higher variance.
* **Statistical learning:**

  * Likelihood $L(\theta|D)=P(D|\theta)$.
  * **MLE:** maximize $L$.
  * **MAP:** maximize $L\times P(\theta)$.
  * Posterior $P(\theta|D)\propto P(D|\theta)P(\theta)$.

---

## 4️⃣ Rational Decision Theory

### 🎲 Lotteries

$L=[p,A;(1-p),B]$ → outcome A with p, B otherwise.
**Expected Utility:** $EU(L)=\sum_i p_iU(x_i)$.
**MEU rule:** pick action with max EU.

### 💰 Utility of Money

* Money ≠ utility → diminishing returns → **risk-averse** (concave U).
* **Risk types:**

  | Shape   | Behavior     | Example U(x) |
  | ------- | ------------ | ------------ |
  | Concave | Risk-averse  | log x, √x    |
  | Linear  | Risk-neutral | x            |
  | Convex  | Risk-seeking | x²           |
* **Certainty Equivalent:** sure x s.t. U(x)=EU(L).
* **Affine invariance:** U′=k₁U+k₂ (k₁>0) ⇒ same decisions.

### 🧩 Rationality Axioms

1. **Completeness:** can compare any two outcomes.
2. **Transitivity:** if A > B and B > C ⇒ A > C.
3. **Continuity:** mixtures possible.
4. **Independence:** common components don’t flip preference.
   Violating transitivity ⇒ *money-pump* paradox.

### 🧮 Multiattribute Utility

* $U(x_1,…,x_n)$ when outcomes have many attributes.
* Use **preference independence** to simplify:

  * Additive form $U=\sum w_i u_i(x_i)$
  * Multiplicative form $U=\prod (1+k,u_i)$.

### 📈 Stochastic Dominance

* **First-order:** one distribution’s CDF always below another → dominates.
* If A dominates B, every rational (monotonic U) prefers A.
* Used for comparing uncertain options or causal influences ( + / – arrows ).

---

## 5️⃣ Summary Mind-Map

* **Probability model → Inference:** enumerate / eliminate / sample.
* **Temporal model → Belief update:** forward filtering.
* **Learning → Fit parameters:** MLE/MAP.
* **Decision → Choose action:** maximize expected utility.

---

### ✅ Quick exam steps

1. Identify **given, query, evidence**.
2. For Bayes net → eliminate hidden only.
3. For DT → compute H & IG, pick max.
4. For HMM → apply Elapse→Observe→Normalize.
5. For decisions → compute EU, compare.
6. For learning → compute L, find θ̂ (MLE / MAP).

---



# 🧠 ML & Decision Tree — Quick Review Notes

## 🌳 CART (Classification and Regression Trees)

* Binary tree: each node splits data into **two subsets** $D₁ ∩ D₂ = ∅, D₁ ∪ D₂ = D$.
* Split chosen by **one variable + threshold**; repeated recursively until a **leaf** (terminal node).
* Same attribute can appear multiple times in deeper nodes.
* Works for:

  * **Classification** → uses **Entropy** or **Gini index**
  * **Regression** → minimizes **SSE (Sum of Squared Errors)**

### 🔍 Greedy Split Strategy

* Choose the split with **maximum information gain** (most purity).
* **Recursive Binary Splitting** stops when nodes are pure or cannot split further.
* **Purity:** node has only one class of samples

### 🧮 Impurity Measures

| Measure        | Formula Idea           | Range | Interpretation                    |
| -------------- | ---------------------- | ----- | --------------------------------- |
| **Entropy**    | $-\sum p_i \log_2 p_i$ | 0–1   | 0 = pure, 1 = mixed               |
| **Gini Index** | $1 - \sum p_i^2$       | 0–0.5 | 0 = pure, 0.5 = worst for 2-class |

---

## ⚠️ Why a Single Tree Fails

* Overfits — too sensitive to outliers and imbalanced data.
* High **variance**: small training changes → large structure changes.
* Poor generalization to test data

---

## 🤝 Ensemble Methods

### 🪵 Bagging (Bootstrap Aggregating)

* Train **many models** on different **bootstrapped samples** (sampling *with replacement*).
* Each model votes → **majority (classification)** or **average (regression)**.
* Goal: reduce **variance**, smooth boundaries, and reduce overfitting.
* Example: 30 trees → smoother, more stable than 1 tree (Train acc 1.0, Test acc ↑0.778→0.822).

### 🌲 Random Forest

* **Bagging + Random Attribute Splits**

  * Bootstrapped samples + at each node, pick a **random subset of features** to split.
* Encourages **diversity among trees**, lowering correlation between them.
* Still aggregates via majority/average voting.

**Summary Difference:**

| Method        | Sampling             | Feature Randomness              | Goal                          |
| ------------- | -------------------- | ------------------------------- | ----------------------------- |
| Bagging       | Bootstrapped samples | All features used               | Reduce variance               |
| Random Forest | Bootstrapped samples | Random feature subset per split | Reduce variance + correlation |

---

## ⚙️ Linear vs Non-Linear Classifiers

* **Linear:** decision boundary is a straight line/plane.

  * $w^T x > b → +$, else $–$.
  * Easy to interpret but limited flexibility.
* **Non-Linear:** use transformations (kernels, neural nets) to separate complex data.

  * Example: project to higher dimension where classes become linearly separable.
* **Projection matrix (w):** projects data to simpler decision space.

---

## 🧩 Parametric vs Non-Parametric

| Type               | Description                                 | Examples                        |
| ------------------ | ------------------------------------------- | ------------------------------- |
| **Parametric**     | Fixed #parameters; model form predetermined | Linear Regression, LDA, DNN     |
| **Non-Parametric** | Model complexity grows with data            | Decision Tree, KNN, SVM(kernel) |

---

## 🧠 KNN (K-Nearest Neighbors)

* **Instance-based**, non-parametric: no explicit model.
* Classify by majority vote of nearest K neighbors (distance-based).
* Sensitive to scale, irrelevant features, and high dimensions (curse of dimensionality).

---

## 🔁 HMM vs MM (context for your quiz)

* **Markov Model (MM):** states observable, uses transition $P(W_t|W_{t-1})$.
* **Hidden Markov Model (HMM):** states hidden, uses both

  * Transition: $P(W_t|W_{t-1})$
  * Emission: $P(O_t|W_t)$.
* Solved via **Forward Algorithm** (predict → update), e.g. $P(W_2|O_1,O_2)$ as in your weather example.

---

### 🧮 Forward Algorithm (Steps)

1. **Initialize:** prior $P(W_0)$
2. **Elapse (Predict):** multiply by transition matrix
3. **Observe (Update):** multiply element-wise by emission prob.
4. **Normalize**
   → repeat for all observations.
   Final normalized vector = posterior probability of states.

---


