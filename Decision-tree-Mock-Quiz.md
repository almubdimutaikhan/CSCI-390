

# ✅ **AI Quiz Practice — 25 Questions with Answers**

---

## **1. Learning Agents**

### **Q1.** Name the four components of a learning agent.

**A:** Performance element, Learning element, Critic, Problem generator.

---

### **Q2.** What does the performance element do?

**A:** Chooses actions in the environment based on the current knowledge/policy.

---

### **Q3.** What is the purpose of the critic?

**A:** Provides feedback by evaluating the agent’s performance relative to a standard.

---

### **Q4.** Why is the problem generator needed?

**A:** Encourages exploration by suggesting actions that gather informative experience.

---

### **Q5.** What is supervised learning vs reinforcement learning?

**A:**

* Supervised: agent receives explicit correct labels.
* Reinforcement: agent receives reward/punishment, not explicit labels.

---

## **2. Ockham’s Razor & Hypotheses**

### **Q6.** State Ockham’s Razor in ML terms.

**A:** Prefer the simplest hypothesis that fits the data; simpler models generalize better.

---

### **Q7.** Why can a very large hypothesis space cause overfitting?

**A:** Because it contains many complex hypotheses that can perfectly fit noise.

---

### **Q8.** For `n` Boolean attributes, how many Boolean functions exist?

**A:** `2^(2^n)`.

---

### **Q9.** For purely conjunctive hypotheses over `n` attributes, how many hypotheses exist?

**A:** `3^n` (attribute included positive / included negated / not included).

---

### **Q10.** What does it mean for a hypothesis to be *consistent*?

**A:** It correctly classifies all training examples.

---

## **3. Decision Tree Learning (DTL)**

### **Q11.** What is the purpose of entropy in decision trees?

**A:** Measures impurity/uncertainty in a dataset.

---

### **Q12.** Write the entropy formula.

**A:** `H = − p*log2(p) − q*log2(q)` where p and q are class proportions.

---

### **Q13.** What is Information Gain?

**A:** `IG(A) = H(parent) − Remainder(A)`.

---

### **Q14.** When does DTL stop?

**A:**

* All examples same class
* No attributes left
* No examples left

---

### **Q15.** Why do we remove an attribute after splitting on it?

**A:** Because it’s already known along that path; retesting gives zero information.

---

## **4. Entropy & IG Calculations**

### **Q16.** Parent has 8 Yes and 8 No. What is entropy?

**A:** `H = 1.0 bit` (perfectly balanced).

---

### **Q17.** If splitting on A yields branch entropies `[0.0, 0.5, 0.0]` with weights `[0.25, 0.5, 0.25]`, compute Remainder.

**A:** `Remainder = 0.25*0 + 0.5*0.5 + 0.25*0 = 0.25`.

---

### **Q18.** Using the previous values, if parent entropy = 1.0, compute IG.

**A:** `IG = 1.0 − 0.25 = 0.75`.

---

### **Q19.** What kind of attribute yields high IG?

**A:** One that splits data into pure (low entropy) subsets.

---

### **Q20.** Why can early stopping fail?

**A:** It may miss important attribute combinations (e.g., XOR requires deeper splits).

---

## **5. Overfitting & Validation**

### **Q21.** What is overfitting?

**A:** Model fits training data too closely, capturing noise; fails on new data.

---

### **Q22.** How does cross-validation help?

**A:** Reduces variance in performance estimates and supports better model selection.

---

### **Q23.** Does bagging reduce bias or variance for a decision tree?

**A:** Variance.

---

### **Q24.** Why does a deeper tree typically overfit?

**A:** It memorizes specific patterns in small subsets of data, including noise.

---

## **6. Chi-Square Pruning**

### **Q25.** What is the null hypothesis in chi-square pruning?

**A:** The attribute being tested does **not** influence class distribution; observed differences are due to chance.

---


