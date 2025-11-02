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

[
H(Z) = -\sum_i p_i \log_2 p_i
]

* Measures **uncertainty** in Z.
* Example: 5 True, 5 False → (H(Z)=1).

---

### 2️⃣ Step 2 — Compute Conditional Entropy for each Attribute

For each attribute (A):

* Split the data by A’s values (e.g., A=0 and A=1).
* For each subset, compute entropy of Z inside it.

[
H(Z|A) = \sum_{v \in Values(A)} P(A=v) , H(Z|A=v)
]

where
[
H(Z|A=v) = -\sum_i P(Z=i | A=v) \log_2 P(Z=i | A=v)
]

✅ Weighted average:
Each subset entropy is weighted by its proportion in total data.

---

### 3️⃣ Step 3 — Compute Information Gain

[
IG(Z, A) = H(Z) - H(Z|A)
]

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

[
\begin{aligned}
H(Z) &= -\sum p_i \log_2 p_i \
H(Z|A) &= \sum_v P(A=v) H(Z|A=v) \
IG(Z,A) &= H(Z) - H(Z|A)
\end{aligned}
]

---

### ✅ Tip for Fast Exam Solving

1. Count positives and negatives for Z.
2. Compute base entropy.
3. Split by each attribute.
4. Compute branch entropies → weighted sum.
5. Pick the one with **max IG**.

---

