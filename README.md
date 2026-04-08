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
├── 📂 Day-02-Optimization-Mastery/             # Advanced Training & Regularization
│   ├── 📂 Dataset/                             # Local CSV Storage
│   │   └── data.csv                            # Clean Diagnostic dataset
│   ├── 📂 Masterclass-Core-Concepts/           # Deep Dive: Optimization Theory
│   │   └── DAY2_MasterClass.ipynb              # L2, Early Stopping & Adam Theory
│   └── 📂 Projects/                            # Practical Applications
│       ├── Clinical_AI_Pipeline_Regularization_Study.ipynb 
│       └── Neural_Noise_Filter.ipynb           # Overfitting & Boundary Visuals
│
|── 📂 Day-03-NLP-Foundations/                  # Text Intelligence & Vectorization
|    ├── 📂 Dataset/                             # Corpus Storage
|    │   ├── IMDB Dataset.csv                    # Sentiment Analysis Data
|    │   └── 📂 Dataset for Multi-Class Emotion Recognition/
|    │       ├── train.txt                       # Training Corpus
|    │       ├── test.txt                        # Final Evaluation Set
|    │       └── val.txt                         # Validation Set
|    ├── 📂 Masterclass-Core-Concepts/           # NLP Pipelines
|    │   └── NLP_DAY3_Text_Preprocessing.ipynb   # Cleaning, Tokenization & TF-IDF
|    └── 📂 Projects/                            # Applied NLP
|        ├── Multi_Class_Emotion_Detector.ipynb  # 6-Class Human Emotion Recognition
|       └── SMS_Spam_Shield.ipynb               # Bayesian Pattern Classification
│
└── 📂 Day-04-Computer-Vision/                 # CNNs & Visual Intelligence
    ├── 📂 Dataset/                            # Visual Data Storage
    │   └── mnist_test.csv                     # Digit benchmarking set
    ├── 📂 MasterClass-Concepts/               # Deep Dive: Spatial Hierarchies
    │   └── DAY_4_CNN_MNIST.ipynb              # Kernels, Pooling & Convolution Theory
    └── 📂 Projects/                           # Applied Vision Projects
        ├── Neural_X_Ray_Analysis/             # Feature Map Interpretability
        └── Real_World_Color_Object_Recognition/ # CIFAR-10 Engine (RGB)
```

# 🚀 Day 1: The Foundation Phase

## 🧠 Neural Architecture from Scratch
In this module, I implemented a fully connected neural network using only Matrix Calculus.

- **XOR Benchmark:** Proved the necessity of hidden layers for non-linear separability.
- **Clinical Pipeline:** Transferred scratch logic to a Breast Cancer Diagnostic task.
- **Visuals:** Used Correlation Heatmaps and Violin Plots to identify dominant predictors for malignancy.

---

# 🔥 Day 2: Optimization & Reliability

## 🛡️ Regularization Masterclass
Focusing on generalization to move from "Memorization" to "Intelligence".

- **Techniques:** L2 Regularization (Weight Decay), Early Stopping, and Adam Optimizer.
- **Insights:** Adaptive learning rates significantly reduced sensitivity to poor weight initialization.
- **Metrics:** Prioritized Recall to ensure zero False Negatives in clinical diagnosis.

---

# 📝 Day 3: NLP Foundations & Text Intelligence

## 🌐 Bridging Human Language and Machine Logic
Day 3 focused on converting raw human text into structured mathematical vectors.

---

# 🖼️ Day 4: Computer Vision & Visual Intelligence

## 🌐 Transitioning to Spatial Data Hierarchies

In Day 4, I bridged the gap between linear data and **Spatial Intelligence** by implementing **Convolutional Neural Networks (CNNs)**.

---

## 🧪 Vision Engineering Excellence

| Module              | Technique            | HKS Engineering Rationale |
|--------------------|--------------------|--------------------------|
| Feature Extraction | Conv2D Kernels     | Capturing local patterns (edges, textures) instead of raw pixels |
| Downsampling       | MaxPooling         | Achieving translation invariance and reducing computational load |
| Stability          | Batch Normalization| Stabilizing internal covariate shift during deep training |
| Interpretability   | Activation Maps    | Visualizing hidden layers to prove hierarchical learning (Lines → Shapes) |

---

## 🛠️ Applied Computer Vision Projects

### 🎯 Real-World Color Recognition (CIFAR-10)
Engineered a deep CNN architecture to classify **10 object categories** in high-variance RGB image data.

### 🔬 Neural "X-Ray" Analysis
Developed a visualization engine to inspect intermediate layers, demonstrating how CNNs learn progressively:
> Edges → Patterns → Shapes → Objects

---

## 🛠️ Technical Stack & Skills

### 📚 Specialized Domains
- **Computer Vision:** CNNs, Feature Mapping, Padding/Stride Logic, Pooling  
- **Natural Language Processing:** Tokenization, Lemmatization, TF-IDF, Naive Bayes  
- **Deep Learning Optimization:** Adam, SGD with Momentum, L2 Regularization, Early Stopping  
- **Foundations:** Matrix Calculus, Backpropagation, Activation Functions  

---

## 📊 Visualization Toolkit

### 🔍 Spatial Visualizations
- Neural Activation Heatmaps (Viridis / Magma)
- Filter & Feature Maps

### 📈 Statistical Visualizations
- Confusion Matrix  
- Precision-Recall Reports  
- Loss & Accuracy Curves  

### 📝 NLP Visualizations
- WordClouds  
- TF-IDF Distribution Plots  

---

## 📝 Roadmap: The Path Ahead

| Status | Phase                         | Target Concept |
|--------|------------------------------|----------------|
| ✅     | Day 1: Foundations           | Scratch Neural Networks & XOR Logic |
| ✅     | Day 2: Optimization Mastery  | Adam, Regularization & Early Stopping |
| ✅     | Day 3: NLP Foundations       | Text Preprocessing & Vectorization |
| ✅     | Day 4: Computer Vision       | CNNs, CIFAR-10 & Neural X-Ray |
| ⏳     | Day 5: Sequence Models       | RNN, LSTM & GRU (Temporal Intelligence) |

---

## 💡 Quote

> "A model that memorizes is a database; a model that generalizes is intelligence."

---

<div align="center">

© 2026 **Hemant Sharma (HKS)**. All Rights Reserved.

</div>
