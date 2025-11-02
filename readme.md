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
