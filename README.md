# 🧠 Neural Networks from First Principles (HKS Deep Learning Series)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

Welcome to my structured journey into the heart of Artificial Intelligence. This repository documents my evolution from building Neural Architectures from Scratch using Pure NumPy to mastering the complex world of Deep Learning Optimization..

---

## 👨‍💻 Author Profile
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #e9ecef;">
    <h3 style="margin-top: 0; color: #2c3e50;">Hemant Sharma (HKS)</h3>
    <p style="color: #34495e; margin-bottom: 5px;"><b>Computer Science & Engineering Student</b> | PIET</p>
    <p style="color: #576574; font-size: 14px;">Focused on <b>Machine Learning</b>, <b>Neural Architectures</b>, and <b>Deep Learning Optimization</b>.</p>
    <div style="margin-top: 10px;">
        <a href="https://www.linkedin.com/in/artisthks" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
        <a href="mailto:artist.hks.dev@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white" alt="Email"></a>
    </div>
</div>

---

## 📂 Directory Structure

```text
.
├── 📂 Day-01-The-Foundations/                  # Pure NumPy Implementation
│   ├── 📂 Dataset/                             # Clinical Data Storage
│   │   └── breast-cancer.csv                   # Raw Diagnostic dataset
│   ├── 📂 Foundations/                         # Core Mathematical Logic
│   │   └── Neural-Networks.ipynb               # XOR Problem & Backprop Engine
│   └── 📂 Projects/                            # Real-world Implementations
│       ├── Cancer-Standard-Model.ipynb         # Benchmark (Sklearn Data)
│       └── Cancer-Predictions-CSV.ipynb        # End-to-End Pipeline (CSV Data)
│
└── 📂 Day-02-Optimization-Mastery/             # Advanced Training & Regularization
    ├── 📂 Dataset/                             # Local CSV Storage
    │   └── data.csv                            # Clean Diagnostic dataset
    ├── 📂 Masterclass-Core-Concepts/           # Deep Dive: Optimization Theory
    │   └── DAY2_MasterClass.ipynb              # L2, Early Stopping & Adam Theory
    └── 📂 Projects/                            # Practical Applications
        ├── Clinical_AI_Pipeline_Regularization_Study.ipynb 
        └── Neural_Noise_Filter.ipynb           # Overfitting & Boundary Visuals
```

# 🚀 Day 1: The Foundation Phase

## 🧠 Neural Architecture from Scratch

In the Foundations module, I designed and implemented a fully connected neural network from scratch using pure **Matrix Calculus**, without relying on high-level frameworks.

### 🔧 Core Implementation

| Component            | Description |
|---------------------|------------|
| Forward Propagation | Multi-layer linear transformations combined with non-linear activations |
| Activation Functions | Sigmoid, ReLU, Tanh, Leaky ReLU (with manual derivatives) |
| Backpropagation     | Gradient computation using Chain Rule |
| Learning Objective  | Binary classification with non-linear separability |

### 🧪 The XOR Benchmark
- Successfully trained the network on the **XOR problem**
- Demonstrated learning of **non-linear decision boundaries**
- Validated the importance of **hidden layers** in deep learning  

---

## 🔬 Breast Cancer Diagnostic Pipeline

Extended the custom neural network to a real-world **clinical classification task**.

### 📊 Data Processing

| Step              | Description |
|------------------|------------|
| Data Ingestion   | Manual CSV loading and parsing |
| Cleaning         | Removed ID columns and handled null values |
| Encoding         | Converted diagnosis labels into numerical format |

### 📈 Exploratory Data Analysis (EDA)
- Feature relationships analyzed using **Correlation Heatmaps**  
- Distribution of nuclear features visualized via **Violin Plots**  
- Identified dominant predictors for malignancy detection  

### 🎯 Clinical Performance
- Focused on minimizing **False Negatives**  
- Achieved high precision and reliable classification  
- Ensured model suitability for **sensitive medical applications**  

---

# 🔥 Day 2: Optimization & Reliability

## 🛡️ Regularization & Optimization Masterclass

Day 2 focused on strengthening model generalization and ensuring robustness for real-world deployment.

---

## ⚙️ Optimization & Regularization Overview

| Technique            | Purpose | Outcome |
|---------------------|--------|--------|
| L2 Regularization   | Penalize large weights | Reduced overfitting |
| Early Stopping      | Stop training at optimal point | Prevented performance degradation |
| Adam Optimizer      | Adaptive learning rates + momentum | Faster convergence |
| SGD (Baseline)      | Standard gradient descent | Used for comparison |

---

## 🧱 Key Concepts Implemented

### 1️⃣ The Regularization Shield
- Applied **L2 Regularization (Alpha)** to control model complexity  
- Integrated **Early Stopping** based on validation loss  
- Shifted model behavior from memorization → generalization  

---

### 2️⃣ Adaptive Optimization (Adam vs SGD)
- Studied internal working of **Adam Optimizer**
- Compared against **SGD baseline**

#### 🔍 Insights:
- Adaptive learning rates improved training stability  
- Momentum accelerated convergence across complex loss surfaces  
- Reduced sensitivity to poor initialization  

---

### 3️⃣ Clinical Research Pipeline

#### 🔍 High-Fidelity EDA
- Implemented **Lower Triangle Heatmaps**  
- Eliminated redundant correlations for clarity  

#### 📊 Decision Boundaries
- Visualized effect of **alpha values (λ)**  
- Observed smoother, generalized boundaries with regularization  

#### 🎯 Recall-First Metrics

| Metric        | Importance |
|--------------|----------|
| Recall       | Minimizes false negatives (critical in diagnosis) |
| ROC-AUC      | Measures overall classification capability |

- Prioritized **Sensitivity (Recall)** to ensure no malignancy goes undetected  

---

## 🛠️ Technical Stack & Skills

### 📚 Core Foundations

| Domain          | Concepts |
|----------------|---------|
| Mathematics    | Linear Algebra, Chain Rule |
| Regularization | L2 Penalty Logic |
| Optimization   | Gradient Descent, Adam |

---

### ⚡ Advanced Training Techniques

- Early Stopping  
- He Initialization  
- Xavier Initialization  
- Adaptive Optimization (Adam)  

---

### 📊 Visualization Toolkit

- Correlation Heatmaps  
- Loss Curves  
- Decision Boundary Plots  

---

### 🧰 Libraries Used

| Category        | Tools |
|----------------|------|
| Numerical      | NumPy |
| Data Handling  | Pandas |
| Visualization  | Matplotlib, Seaborn |
| ML Utilities   | Scikit-learn |

---

## 📝 Roadmap: The Path Ahead

| Status | Phase |
|------|------|
| ✅ | Day 1: Scratch Neural Network & Binary Classification |
| ✅ | Day 2: Optimization, Regularization & Overfitting Prevention |
| ⏳ | Day 3: Computer Vision (MNIST Handwritten Digits) |
| ⏳ | Day 4: Multi-class Classification & Advanced Metrics |

---

## 💡 Quote

> "A model that memorizes is a database; a model that generalizes is intelligence."

---

<div align="center">

**© 2026 Hemant Sharma (HKS). All Rights Reserved.**

</div>
