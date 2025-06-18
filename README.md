# MLPC 2024 Project: Speech Command Recognition for Smart Home Systems  
*(Johannes Kepler University Linz — Erasmus Program, Spring 2024-2025)*

This repository contains my complete project work for the **MLPC 2024 Project (Machine Learning Practical Course)** at **JKU Linz**.  
The project focused on developing an **on-device speech command recognition system** for a fictional Austrian company, **SmartVoiceControl**. The system is designed for **German-speaking smart home users** and aims to control devices like lights, heating, or alarms using voice commands — without needing an activation phrase such as "Hey Siri" or "Ok Google."

---

## 🎯 Project Context

### The Challenge

✅ **SmartVoiceControl** required a solution that:
- **Listens continuously in the background**, with no activation keyword.
- **Runs locally on resource-constrained devices** (phones, watches, tablets) for privacy and efficiency.
- **Minimizes battery and computation load** (lightweight model).
- **Delivers high accuracy** with **low false positives/negatives** to ensure reliability in everyday use.
- **Handles speaker variability** (native and non-native German speakers, Austrian dialects, various devices).

---

## 📚 Project Phases

The project was carried out in **four structured phases** to reflect a real-world machine learning pipeline:

---

### 🟣 **Task 1: Data Collection**
- Recorded **240 speech samples** containing German keywords (`"Licht"`, `"Radio"`, `"an"`, `"aus"`, etc.).
- Captured **5 realistic smart home acoustic scenes** (10-30 sec) containing ambient noise, speech commands, and distractor speech.
- Ensured quality and diversity of recordings (different rooms, devices, and acoustic conditions).

📁 *Outputs:*  
- Raw audio files (cleaned and annotated).  
- Documentation of recording setup and conditions.

---

### 🟣 **Task 2: Data Exploration**
- Analyzed pre-computed **audio feature sets** (MFCCs, chroma features, spectral features, etc.).
- Visualized distributions, correlations, and class separability using:
  - Histograms, scatter plots, PCA projections.
- Identified features that best separate keywords from noise/distractors for classifier training.

📁 *Outputs:*  
- Data exploration notebooks.  
- Feature selection rationale.  
- Summary report with visualizations and conclusions.

---

### 🟣 **Task 3: Classification**
- Implemented and evaluated multiple classifiers:
  - **Logistic Regression**
  - **Support Vector Machines (SVM)**
  - **Random Forests**
  - **Gradient Boosting**
- Created balanced data splits: train, validation, test.
- Performed **hyperparameter tuning** (grid search, cross-validation).
- Designed **custom cost-sensitive objectives** to reduce costly misclassifications (e.g., false alarm deactivation).

📁 *Outputs:*  
- Classifier code and training logs.  
- Performance reports (accuracy, precision, recall, F1, cost).  
- Hyperparameter search results.

---

### 🟣 **Task 4: Final Challenge**
- Applied tuned classifiers to realistic smart home scenes.
- Used a **provided cost matrix** to quantify penalties for:
  - False positives (e.g., switching off the alarm unintentionally).
  - False negatives (e.g., failing to turn on heating).
- Selected final model and configuration to minimize total cost on blind test set.

📁 *Outputs:*  
- Final predictions for test set submission.  
- Cost evaluation report.  
- Summary of model selection process.

---

## ⚙️ Tools & Libraries

- **Python 3.x**
- **NumPy / SciPy** — Numerical operations
- **scikit-learn** — Machine learning algorithms and utilities
- **matplotlib / seaborn** — Visualization
- **Jupyter / Google Colab** — Interactive development

Optional:
- **Librosa** — Audio feature extraction (if custom features were added)


