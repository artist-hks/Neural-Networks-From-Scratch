<div align="center">

# 🧠 Neural Networks From Scratch



![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Workshop-Completed-brightgreen?style=for-the-badge)
![Days](https://img.shields.io/badge/Duration-5%20Days-blue?style=for-the-badge)

<br/>

**5-Day Intensive Deep Learning Workshop · Upflairs Pvt. Ltd.**

*A hands-on journey from building neural networks with pure NumPy to deploying CNNs, NLP pipelines, and Sequence Models.*

[📅 Workshop Overview](#-workshop-overview) · [📂 Repo Structure](#-repository-structure) · [🗓️ Day-by-Day](#-day-by-day-breakdown) · [🚀 Quick Start](#-quick-start)

</div>

---

## 📌 About This Repository

This repository is the complete documented journey of a **5-Day Neural Networks Workshop** by **Upflairs Pvt. Ltd.** Each day has two components: a **MasterClass notebook** covering core theory and implementation, and **hands-on Projects** applying those concepts to real-world problems.

| | |
|---|---|
| 🏫 **Organizer** | Upflairs Pvt. Ltd. |
| 🧑‍🎓 **Participant** | Hemant Sharma ([@artist-hks](https://github.com/artist-hks)) |
| 📅 **Format** | 5-Day Intensive — Theory + Projects every day |
| 🛠️ **Tools** | Python, NumPy, TensorFlow/Keras, Scikit-Learn, NLTK, Jupyter |
| 📈 **Progression** | Pure NumPy Neural Nets → Regularization → NLP → CNNs → RNN/LSTM/GRU |

---

## 🗺️ Workshop Overview

```
DAY 1              DAY 2              DAY 3              DAY 4              DAY 5
─────────          ─────────          ─────────          ─────────          ─────────
Neural Nets        The Science        NLP &              Computer           Sequence
From Scratch       of Learning        Text AI            Vision             Models
(Pure NumPy)       (Optimisation,     (Preprocessing,    (CNNs,             (RNN, LSTM,
                   Regularisation,    TF-IDF, NLP        MNIST,             GRU, Time
                   Initialisation)    Classifiers)       CIFAR-10)          Series)

PROJECT:           PROJECT:           PROJECT:           PROJECT:           PROJECT:
Breast Cancer      Clinical AI        SMS Spam           Neural X-Ray       Quantum
Detection          Pipeline +         Shield +           Analysis +         Forecaster +
(2 variants)       Noise Filter       Emotion            Color Object       Cyber
                                      Detector           Recognition        Sentinel
```

---

## 📂 Repository Structure

```
📦 Neural-Networks-From-Scratch/
│
├── 📁 DAY 1/
│   ├── 📁 Foundation/
│   │   └── 📓 Neural_Networks.ipynb              ← Core theory: XOR, forward/backprop from scratch
│   ├── 📁 Projects/
│   │   ├── 📓 Breast_Cancer_Detection_Standard_Model.ipynb
│   │   └── 📓 Breast_Cancer_Analysis_Neural_Predictions_CSV.ipynb
│   └── 📁 Dataset/
│       └── 📊 breast-cancer.csv
│
├── 📁 DAY 2/
│   ├── 📁 MasterClass Core Concepts/
│   │   └── 📓 DAY_2_MasterClass.ipynb            ← Optimisers, Initialisation, Regularisation
│   ├── 📁 Projects/
│   │   ├── 📓 Clinical_AI_Pipeline_Regularization_Study.ipynb
│   │   └── 📓 Neural_Noise_Filter.ipynb
│   └── 📁 Dataset/
│       └── 📊 data.csv
│
├── 📁 DAY 3/
│   ├── 📁 MasterClass Concepts/
│   │   └── 📓 NLP_DAY3_Text_Preprocessing.ipynb  ← Cleaning, Tokenization, TF-IDF
│   ├── 📁 Projects/
│   │   ├── 📓 SMS_Spam_Shield.ipynb
│   │   └── 📓 Multi_Class_Emotion_Detector.ipynb
│   └── 📁 Dataset/
│       ├── 📊 IMDB Dataset.csv
│       └── 📁 Dataset for Multi-Class Emotion Recognition/
│           ├── train.txt
│           ├── test.txt
│           └── val.txt
│
├── 📁 DAY 4/
│   ├── 📁 MasterClass Concepts/
│   │   └── 📓 DAY_4_CNN_MNIST.ipynb              ← CNN architecture, Conv2D, MaxPooling, MNIST
│   ├── 📁 Projects/
│   │   ├── 📓 Real_World_Color_Object_Recognition.ipynb
│   │   └── 📓 Neural_X_Ray_Analysis.ipynb
│   └── 📁 Dataset/
│       └── 📊 mnist_test.csv
│
├── 📁 DAY 5/
│   ├── 📁 MasterClass Concepts/
│   │   └── 📓 Day_5_Sequence_Models.ipynb        ← RNN vs LSTM vs GRU benchmark
│   └── 📁 Projects/
│       ├── 📓 Quantum_Forecaster.ipynb
│       ├── 📓 Cyber_Sentinel.ipynb
│       └── 📄 Cyber_Sentinel Report.pdf
│
└── 📄 README.md
```

---

## 🗓️ Day-by-Day Breakdown

---

### 📗 Day 1 — Neural Networks From First Principles

> *Building a fully working neural network using only NumPy — no frameworks.*

**MasterClass: `Neural_Networks.ipynb`**

The first day started from absolute scratch. A custom `NeuralArchitect` class was implemented in pure NumPy, with full forward propagation and backpropagation coded by hand. The XOR problem was used as the benchmark to verify non-linear learning.

**Concepts Covered:**
- Neurons, weights, biases and the forward pass: `Z = W·A + b`
- Activation functions implemented from scratch: **Sigmoid, Tanh, ReLU, Leaky ReLU** (with their derivatives)
- **Binary Cross-Entropy Loss** and its gradient
- **Backpropagation** via the chain rule: computing `dW`, `db` layer by layer
- **He Initialization** for stable gradient flow
- Training loop and loss visualization

**Projects:**

| Project | Dataset | Task | Key Technique |
|---------|---------|------|---------------|
| 🩺 **Breast Cancer Detection** (Standard) | `sklearn` breast cancer | Binary Classification | Custom `NeuralArchitect` [30→16→1] |
| 🩺 **Breast Cancer Analysis** (CSV Pipeline) | `breast-cancer.csv` | Binary Classification + Full EDA | Raw CSV → Cleaning → EDA → Neural Prediction |

The CSV variant adds a complete data engineering pipeline: dropping irrelevant columns, encoding the `M/B` diagnosis target, feature correlation heatmaps, and StandardScaler normalization before training.

📁 `DAY 1/`

---

### 📘 Day 2 — The Science of Learning

> *Why your network fails to converge — and exactly how to fix it.*

**MasterClass: `DAY_2_MasterClass.ipynb`**

Day 2 moved beyond just building a network to understanding *why* certain training choices matter. Three critical pillars were explored: how you **scale** data, how you **initialize** weights, and how you **regularize** the model.

**Concepts Covered:**
- **Feature Scaling** with `StandardScaler` — why unscaled features cause unbalanced gradients
- **Optimiser Comparison** — SGD (slow) vs Momentum (faster) vs **Adam** (adaptive, best)
- **Weight Initialization** — All Zeros (broken), Large Random (exploding), **He/Xavier** (correct)
- **Regularisation** — L2 penalty (`alpha`) to prevent memorization
- **Early Stopping** — halting training when validation plateaus
- Confusion matrix and ROC curve evaluation

**Projects:**

| Project | Dataset | Task | Key Technique |
|---------|---------|------|---------------|
| 🩺 **Clinical AI Pipeline** | Breast Cancer CSV | Overfitting vs Generalisation Study | Comparing 3 models: Overfit / Adam+L2 / Early Stop |
| 🌊 **Neural Noise Filter** | Synthetic Moon Dataset | Decision Boundary Visualisation | Low α (overfit) vs High α (regularized) boundary demo |

The Neural Noise Filter is a pure visual experiment showing exactly how regularization forces the decision boundary from zig-zag chaos to a clean smooth curve.

📁 `DAY 2/`

---

### 📙 Day 3 — NLP & Text Intelligence

> *Teaching machines to read — from raw noisy text to structured predictions.*

**MasterClass: `NLP_DAY3_Text_Preprocessing.ipynb`**

Day 3 introduced Natural Language Processing. The full text preprocessing pipeline was built on the **IMDB movie reviews dataset** (50K reviews), implementing a custom `hks_clean_pipeline` for production-grade text cleaning.

**Concepts Covered:**
- **Text Cleaning Pipeline:** Lowercasing → HTML/URL stripping → Punctuation removal → Stopword filtering
- **Tokenization** with NLTK's `word_tokenize`
- **Lemmatization** using `WordNetLemmatizer` (root extraction)
- **Word Frequency Analysis** and corpus visualization
- **TF-IDF Vectorization:** `TF-IDF(t,d) = TF(t,d) × log(N/DF(t))` — rewarding meaningful words, penalizing common ones

**Projects:**

| Project | Dataset | Task | Model |
|---------|---------|------|-------|
| 🛡️ **SMS Spam Shield** | SMS Spam Collection | Binary Text Classification (Ham/Spam) | TF-IDF + Multinomial Naive Bayes |
| 🎭 **Multi-Class Emotion Detector** | Emotion Recognition Dataset (train/test/val) | 6-class emotion classification | TF-IDF + Multinomial Naive Bayes |

The Emotion Detector classifies text into 6 emotion classes and includes a live `detect_emotion()` function. The Spam Shield includes a `predict_spam()` function for real-time inference.

📁 `DAY 3/`

---

### 📕 Day 4 — Computer Vision with CNNs

> *Teaching machines to see — from pixel grids to object recognition.*

**MasterClass: `DAY_4_CNN_MNIST.ipynb`**

Day 4 introduced Convolutional Neural Networks using TensorFlow/Keras. The architecture was built progressively: Conv2D for feature extraction, MaxPooling for downsampling, Dropout for regularization, and Dense layers for classification — trained on MNIST handwritten digits.

**Concepts Covered:**
- **Convolution:** 2D filters scanning for edges, corners, patterns
- **Feature Maps:** What each Conv layer "sees"
- **MaxPooling2D:** Spatial downsampling, translation invariance
- **BatchNormalization:** Stabilizing activations between layers
- **Dropout:** Randomly disabling neurons to prevent co-adaptation
- **Activation Map Visualization:** Peering inside each conv layer

**Projects:**

| Project | Dataset | Task | Architecture |
|---------|---------|------|--------------|
| 🎨 **Real-World Color Object Recognition** | CIFAR-10 (32×32 RGB, 10 classes) | Multi-class Color Image Classification | Deep CNN: Conv→BN→Pool→Dropout ×3 |
| 🔬 **Neural X-Ray Analysis** | CIFAR-10 | CNN + **Intermediate Activation Visualisation** | CNN + Keras Feature Extractor Model |

The X-Ray Analysis project goes beyond just training — it extracts and visualizes activation maps from each Conv layer, showing exactly what features the network detects at each stage (edges → textures → shapes → objects).

📁 `DAY 4/`

---

### 📓 Day 5 — Sequence Models & Temporal Intelligence

> *Networks that remember — processing sequences, text, and time-series data.*

**MasterClass: `Day_5_Sequence_Models.ipynb`**

The final day covered recurrent architectures: how standard RNNs suffer from vanishing gradients, how LSTMs solve this with gating mechanisms, and how GRUs offer a leaner alternative. All three were benchmarked on the **IMDB sentiment dataset**.

**Concepts Covered:**
- **Recurrent Neural Networks (RNN):** Hidden state loops, vanishing gradient problem
- **LSTM (Long Short-Term Memory):** Forget gate, input gate, output gate — retaining long-range context
- **GRU (Gated Recurrent Unit):** Reset gate, update gate — efficient LSTM alternative
- **Embedding Layer:** Dense word vector representations
- **Bidirectional LSTM:** Reading sequences forward and backward for fuller context
- Architecture comparison: RNN (~80%) vs GRU (~86.5%) vs LSTM (~87% val accuracy)

**Projects:**

| Project | Dataset | Task | Architecture |
|---------|---------|------|--------------|
| 📉 **Quantum Forecaster** | Synthetic multi-component signal | Time-Series Regression | Stacked GRU [64→32] + Dropout |
| 🛡️ **Cyber Sentinel** | IMDB Reviews | Deep Contextual Sentiment Analysis | Bidirectional LSTM + GlobalMaxPool1D |

The Quantum Forecaster predicts a complex synthetic signal combining sine waves, trend, and noise using a stacked GRU regression model. The Cyber Sentinel uses a Bidirectional LSTM to read text in both directions for stronger sentiment understanding, achieving ~88% validation accuracy.

📁 `DAY 5/`

---

## 🛠️ Tech Stack

| Tool | Used For |
|------|----------|
| **Python 3.x** | Core language |
| **NumPy** | From-scratch neural network math (Day 1) |
| **TensorFlow / Keras** | CNN and sequence model training (Days 4–5) |
| **Scikit-Learn** | MLPClassifier, preprocessing, metrics (Days 2–3) |
| **NLTK** | Tokenization, lemmatization, stopwords (Day 3) |
| **Pandas** | Data loading and manipulation |
| **Matplotlib / Seaborn** | Visualizations, training curves, confusion matrices |
| **Jupyter Notebook / Google Colab** | Interactive coding environment |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nltk jupyter
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Run Locally
```bash
git clone https://github.com/artist-hks/Neural-Networks-From-Scratch.git
cd Neural-Networks-From-Scratch
jupyter notebook
```

Navigate to any `DAY X/` folder and open a notebook. Recommended order: **Foundation/MasterClass first, then Projects**.

### Run on Google Colab
Upload any `.ipynb` directly to [colab.research.google.com](https://colab.research.google.com) — no local setup required. Notebooks that use `google.colab.files` or `drive.mount` are designed for Colab.

---

## 📈 Projects at a Glance

| Day | Project | Domain | Model | Outcome |
|-----|---------|--------|-------|---------|
| 1 | Breast Cancer Detection | Healthcare | Custom NumPy NN [30→16→1] | Binary classification from scratch |
| 1 | Breast Cancer Analysis (CSV) | Healthcare | Custom NumPy NN + EDA | Full data pipeline to prediction |
| 2 | Clinical AI Pipeline | Healthcare | MLPClassifier + Regularization | Overfitting vs generalization study |
| 2 | Neural Noise Filter | Synthetic | MLPClassifier | Boundary visualization |
| 3 | SMS Spam Shield | NLP | TF-IDF + Naive Bayes | Ham/Spam classifier |
| 3 | Multi-Class Emotion Detector | NLP | TF-IDF + Naive Bayes | 6-emotion text classification |
| 4 | Color Object Recognition | Computer Vision | Deep CNN | CIFAR-10 10-class classifier |
| 4 | Neural X-Ray Analysis | Computer Vision | CNN + Activation Maps | Model interpretability |
| 5 | Quantum Forecaster | Time Series | Stacked GRU | Signal regression |
| 5 | Cyber Sentinel | NLP | Bidirectional LSTM | ~88% sentiment accuracy |

---

## 💡 Key Learnings

- ✅ Built a fully working neural network from pure NumPy — no frameworks
- ✅ Understood *why* Adam beats SGD and *why* He initialization matters
- ✅ Learned the complete NLP pipeline from raw text to TF-IDF vectors
- ✅ Implemented CNNs with Conv2D, BatchNorm, Dropout and visualized what each layer learns
- ✅ Understood the LSTM gating mechanism and why it handles long sequences better than RNNs
- ✅ Built and compared RNN, LSTM, and GRU on the same task with quantified results

---

## 🔗 Connect

[![Portfolio](https://img.shields.io/badge/Portfolio-artist--hks.vercel.app-black?style=flat-square&logo=vercel)](https://artist-hks.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-artisthks-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/artisthks)
[![GitHub](https://img.shields.io/badge/GitHub-artist--hks-gray?style=flat-square&logo=github)](https://github.com/artist-hks)
[![Email](https://img.shields.io/badge/Email-artist.hks.dev%40gmail.com-red?style=flat-square&logo=gmail)](mailto:artist.hks.dev@gmail.com)
[![LeetCode](https://img.shields.io/badge/LeetCode-artist__hks-orange?style=flat-square&logo=leetcode)](https://leetcode.com/artist_hks)

---

<div align="center">
  <sub>
    Workshop by <strong>Upflairs Pvt. Ltd.</strong> &nbsp;|&nbsp;
    Documented by <a href="https://github.com/artist-hks">Hemant Sharma</a>
  </sub>
  <br/><br/>
  <sub>⭐ If this repo helped you learn something new, drop a star!</sub>
</div>
