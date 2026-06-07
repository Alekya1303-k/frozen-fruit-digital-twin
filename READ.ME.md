# 🍓 Agent-Based Digital Twin Framework for Frozen Fruit Quality Prediction

> **Pace University · MS Data Science · 2026**  
> Alekya Koppaka · Padmaja Preksha Dinesh · Tripsy Bhaveshbhai Kheni

---

## 📌 Overview

Frozen fruit degrades silently — ice crystal formation, freeze-thaw cycles, and freezer burn cause irreversible damage long before any visible signs appear. By the time cold-chain managers notice something is wrong, the loss is already done.

This project proposes an **Agent-Based Digital Twin Framework** that combines fruit image analysis with real-time IoT sensor data to continuously monitor, forecast, and explain frozen fruit quality decline — turning a reactive process into a proactive one.

> This is the **first end-to-end system** combining multimodal Digital Twin fusion, frozen-specific degradation simulation, explainable AI, and agent-based decision making for frozen produce quality management.

---

## 🎯 Problem Statement

- 💸 **$1 trillion** in annual food waste from cold-chain inadequacies worldwide
- 🧊 **30%** of frozen food lost to undetected temperature abuse and freeze-thaw events
- ⚠️ Existing monitoring is **reactive and manual** — damage done before detection
- 📂 **No public dataset** captures frozen-specific degradation visual effects
- 🔗 No system fuses **image + sensor + agent** in one end-to-end pipeline

---

## 🏗️ System Architecture — 5-Module Pipeline

```
Fruit Images ──► Module 1: CNN Classifier (EfficientNet-B0 + Grad-CAM) ──► 1280-dim embedding
                                                                                    │
IoT Sensors ───► Module 2: BiLSTM Forecaster (Bahdanau Attention)      ──►  256-dim embedding
                                                                                    │
                          Module 3: Digital Twin Fusion  ◄──────────────── concat → 1536-dim
                                        │
                          Module 4: Hybrid Agent Decision Module (5-tier alerts)
                                        │
                          Module 5: LLM Dashboard (Claude API · Streamlit · 15-min cadence)
```

---

## 📂 Repository Structure

```
frozen-fruit-digital-twin/
│
├── notebooks/
│   ├── Module1.ipynb          # CNN Image Classifier (EfficientNet-B0 + Grad-CAM)
│   ├── Module_2.ipynb         # BiLSTM Time-Series Forecaster
│   ├── Module3.ipynb          # Digital Twin Fusion
│   ├── Module4.ipynb          # Hybrid Agent Decision Module
│   └── Module5.ipynb          # LLM-Powered Dashboard
│
├── docs/
│   ├── Frozen_fruit_poster_final.pdf        # Research poster
│   ├── Draft_paper.pdf                      # Draft research paper
│   ├── FrozenFruit_Analytical_paper.pdf     # Analytical paper
│   ├── Frozen_fruit_quality_prediction.pdf  # Final paper
│   ├── Deliverable_1.docx                   # Project deliverable
│   └── FrozenFruit_DigitalTwin_midterm.pptx # Midterm presentation
│
├── demo/
│   └── Demo.mp4               # Live dashboard demo video
│
├── requirements.txt
└── README.md
```

---

## 🔬 Modules in Detail

### Module 1 — CNN Image Classifier
- **Architecture:** EfficientNet-B0 with two-phase transfer learning
- Phase 1: Frozen backbone → 72% baseline accuracy
- Phase 2: Fine-tune top 50 layers → **97% test accuracy**
- **Explainability:** Grad-CAM heatmaps on all 3 quality classes
- **Output:** 1280-dim visual embedding

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Fresh | 0.91 | 1.00 | 0.95 |
| Slightly Degraded | 1.00 | 0.90 | 0.95 |
| Spoiled | 1.00 | 1.00 | 1.00 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** |

---

### Module 2 — BiLSTM Time-Series Forecaster
- **Architecture:** BiLSTM(128) + Bahdanau Attention
- **Input:** 30-timestep IoT sensor sequences (Temperature, Humidity, CO₂, Light)
- **Physics layer:** Arrhenius kinetics `k(T) = A·exp(−Ea/RT)` per fruit type
- **Output:** 256-dim temporal embedding + quality score
- **Result:** RMSE < 0.15 — **47% improvement** over mean baseline

| Model | RMSE | Improvement |
|-------|------|-------------|
| Mean baseline | 0.283 | — |
| Simple LSTM | 0.181 | −36.0% |
| GRU | 0.163 | −42.4% |
| **BiLSTM + Attention** | **< 0.15** | **−47.0%** |

---

### Module 3 — Digital Twin Fusion
- **Input:** concat(1280-dim CNN, 256-dim BiLSTM) = **1536-dim Digital Twin state**
- **Architecture:** Two-branch fusion → Dense(512) → Dense(64) → sigmoid → q̂ ∈ [0,1]
- **Result:** RMSE = **0.0567**, R² = **0.9459** (explains 94.6% of variance)

| Configuration | RMSE | vs CNN-only |
|--------------|------|-------------|
| CNN only | 0.121 | baseline |
| BiLSTM only | 0.103 | −14.9% |
| Simple concat | 0.082 | −32.2% |
| **Two-branch (ours)** | **0.0567** | **−53.1%** |

---

### Module 4 — Hybrid Agent Decision Module
Five-tier autonomous alert system:

| Tier | Score | Action |
|------|-------|--------|
| 🟢 Excellent | q̂ > 0.85 | Normal storage |
| 🟩 Good | 0.70–0.85 | Standard monitoring |
| 🟡 Monitor | 0.50–0.70 | Increase inspections |
| 🟠 Warning | 0.30–0.50 | Immediate intervention |
| 🔴 Critical | ≤ 0.30 | Remove from storage |

- 186 samples escalated by confidence scorer (conf < 0.20)
- 6 rapid decline events detected by trend detector
- **Total cost savings: $6,676** across 3,862 samples

---

### Module 5 — LLM-Powered Dashboard
- **Claude Sonnet API** invoked every 15 minutes via zero-shot prompting
- Generates a 4-component plain-English report:
  1. Quality Status
  2. Root Cause Analysis
  3. Recommended Action
  4. Confidence Level
- Rendered on **Streamlit** with forecast charts, active alerts, and Grad-CAM heatmaps

---

## 🧪 Novel Contribution — Arrhenius Frozen Degradation Simulation

No public dataset captures frozen-specific visual degradation. We simulate the **Slightly Degraded** class using physics-based image augmentation:

1. **Ice crystal formation** — Gaussian blur + blue-white frost overlay
2. **Freeze-thaw damage** — Randomized dark elliptical blotches
3. **Freezer burn** — Circular grey desaturated patches

| Class | Images | Source |
|-------|--------|--------|
| Fresh (Class 0) | 7,614 | Real (Kaggle) |
| Slightly Degraded (Class 1) | 4,740 | **Physics-simulated** |
| Spoiled (Class 2) | 5,985 | Real (Kaggle) |
| **Total** | **18,339** | |

---

## 📊 Results Summary

| Module | Metric | Target | Result |
|--------|--------|--------|--------|
| M1 CNN | Accuracy | > 80% | **97% ✅** |
| M2 BiLSTM | RMSE | < 0.15 | **< 0.15 ✅** |
| M3 Fusion | RMSE | < 0.0954 | **0.0567 ✅** |
| M3 Fusion | R² | > 0.90 | **0.9459 ✅** |
| M4 Agent | 5-tier | Active | **Deployed ✅** |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Alekya1303-k/frozen-fruit-digital-twin.git
cd frozen-fruit-digital-twin
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook notebooks/Module1.ipynb
jupyter notebook notebooks/Module_2.ipynb
jupyter notebook notebooks/Module3.ipynb
jupyter notebook notebooks/Module4.ipynb
jupyter notebook notebooks/Module5.ipynb
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

---

## 🛠️ Tech Stack

`Python` `TensorFlow` `Keras` `EfficientNet-B0` `BiLSTM` `Bahdanau Attention`  
`Grad-CAM` `Claude API` `Streamlit` `Pandas` `NumPy` `Matplotlib` `Scikit-learn`

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Research Poster](docs/Frozen_fruit_poster_final.pdf) | Conference-style poster |
| [Final Paper](docs/Frozen_fruit_quality_prediction.pdf) | Full research paper |
| [Analytical Paper](docs/FrozenFruit_Analytical_paper.pdf) | Detailed analysis |
| [Presentation](docs/FrozenFruit_DigitalTwin_midterm.pptx) | Slide deck |
| [Demo Video](demo/Demo.mp4) | Live dashboard walkthrough |

---

## 🔮 Future Work

- Real IoT validation with instrumented frozen storage facilities
- Edge deployment on Raspberry Pi / NVIDIA Jetson
- Additional fruit types and MAP storage conditions
- Reinforcement learning adaptive agent policy

---

## 👥 Authors

**Alekya Koppaka** — alekya130306@gmail.com  
**Padmaja Preksha Dinesh** — padmajapreksha10@gmail.com  
**Tripsy Bhaveshbhai Kheni** — tripsykheni2488@gmail.com  

Pace University, New York · 2026
