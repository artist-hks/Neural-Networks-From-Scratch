# 🧠 Neural Networks from First Principles (HKS Deep Learning Series)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

Welcome to my deep dive into Artificial Intelligence. This repository documents my journey of building and optimizing Neural Network architectures from scratch using **Pure NumPy**, emphasizing mathematical foundations and clinical diagnostic applications.

---

## 👨‍💻 Author Profile
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #e9ecef;">
    <h3 style="margin-top: 0; color: #2c3e50;">Hemant Sharma (HKS)</h3>
    <p style="color: #34495e; margin-bottom: 5px;"><b>Computer Science & Engineering Student</b> | Poornima University</p>
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
└── Day1/
    ├── Dataset/                           # Clinical Data Storage
    │   └── breast-cancer.csv              # Raw Diagnostic dataset
    ├── Foundations/                       # Core Mathematical Logic
    │   └── Neural-Networks.ipynb          # XOR Problem, Activations & Backprop Engine
    └── Projects/                          # Real-world Implementations
        ├── Cancer-Standard-Model.ipynb    # Diagnostic Benchmark (Sklearn Data)
        └── Cancer-Predictions-CSV.ipynb   # End-to-End Pipeline (CSV Data)
```

# 🚀 Day 1: The Foundation Phase

## 🧠 1. Neural Architecture from Scratch

In the Foundations module, I architected a fully connected network using only Matrix Calculus.

- **Forward Propagation:** Multi-layer linear transformations with non-linear mapping.
- **Activation Suite:** Manual implementation of Sigmoid, ReLU, Tanh, and Leaky ReLU with their respective partial derivatives.
- **The XOR Test:** Successfully trained a model to learn non-linear decision boundaries, proving the power of hidden layers.

---

## 🔬 2. Breast Cancer Diagnostic Pipeline

Applied the custom NeuralArchitect engine to solve a high-stakes medical classification problem.

- **Data Engineering:** Manual CSV ingestion, safe cleaning (dropping ID/Nulls), and target encoding.
- **Exploratory Data Analysis (EDA):**  
  - Feature correlations via Heatmaps  
  - Nuclear morphological distributions using Violin Plots
- **Clinical Accuracy:** Minimized False Negatives to ensure reliable malignancy detection, achieving high precision on test data.

---

## 🛠️ Technical Stack & Skills

- **Core Math:** Linear Algebra, Matrix Multiplication, Chain Rule (Calculus)
- **Preprocessing:** Feature Scaling (Standardization), Label Encoding
- **Evaluation:** Binary Cross-Entropy Loss, Confusion Matrices
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn (Preprocessing only)

---

## 📝 Roadmap: The Path Ahead

- [x] Day 1: Scratch Neural Network & Binary Classification  
- [ ] Day 2: Optimization (Adam Optimizer), Regularization (Dropout/L2), and Overfitting Prevention  
- [ ] Day 3: Scaling to Computer Vision (MNIST Handwritten Digits)  
- [ ] Day 4: Multi-class Classification & Advanced Metrics  

---

## 💡 Quote

> "Deep Learning is not magic; it's a combination of Matrix Calculus and Optimization algorithms." 🚀

---

© 2026 Hemant Sharma. All rights reserved.
