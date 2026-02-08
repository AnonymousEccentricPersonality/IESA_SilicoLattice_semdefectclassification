# SEM-HiNet: Hierarchical Industrial SEM Defect Classification

## 🚀 Project Overview

**SEM-HiNet** is a deep learning framework designed for high-precision, high-speed semiconductor defect classification. It utilizes a **Multi-Head Hierarchical Architecture** built on an NXP-optimized **MobileNetV3** backbone.

The project addresses the "needle in a haystack" problem in industrial Scanning Electron Microscope (SEM) imaging by combining intelligent noise filtering (Gatekeeping) with specialized defect expertise (Specialist Heads).

## 🧠 Technical Approach

Our approach utilizes a single shared backbone to extract high-dimensional features, which are then processed by five parallel classification heads.

* **H1 (The Gatekeeper):** Separates valid SEM data (Normal/Defect) from industrial noise and outliers (Bogus).
* **H2 (The Router):** Directs defects into three primary geometric families: Line Defects, Area Defects, or Edge Defects.
* **H3–H5 (The Specialists):** Binary experts trained to distinguish between subtle textures:
  * **H3:** Bridge vs. Open
  * **H4:** CMP vs. Crack
  * **H5:** LER vs. Incomplete Etch

### Key Training Innovations

* **Stratified Partitioning:** Mathematically ensures proportional class distribution across training and validation sets.
* **Weighted Head Loss:** Prioritizes "Expert" heads (H3-H5) with up to 8x weight to resolve leaf-node confusion.
* **QAT (Quantization-Aware Training):** Simulates 8-bit precision during training for the **i.MX 8M Plus NPU**.

## 📊 Performance Metrics

| Head | Level | Metric | Score |
| :--- | :--- | :--- | :--- |
| **H1** | Gatekeeper | Recall | **~99%** |
| **H2** | Router | Accuracy | **100%** |
| **H3-H5** | Specialists | Precision | **>95%** |

## 📂 Repository Structure

```text
.
├── data/                   # Dataset storage (mapped by folder name)
│   ├── normal/             # Valid SEM scans (No defects)
│   ├── bogus/              # Non-SEM images (Noise/Outliers)
│   ├── bridge/             # Line defect: Bridge
│   ├── open/               # Line defect: Open
│   ├── cmp/                # Area defect: CMP
│   ├── crack/              # Area defect: Crack
│   ├── ler/                # Edge defect: Line Edge Roughness
│   └── inc_etch/           # Edge defect: Incomplete Etch
├── models/                 # Model architecture and weights
│   ├── arch.py             # SEMHierarchicalNetV3_QAT class
│   └── sem_final.pth       # Trained QAT weights
├── scripts/                # Utility and training scripts
│   ├── train.py            # Stratified training & QAT loop
│   ├── evaluate.py         # 5-Head confusion matrix report
│   └── export_onnx.py      # NPU-specific conversion script
├── README.md               # Project documentation
└── requirements.txt        # Dependency list
