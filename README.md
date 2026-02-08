SEM-HiNet: Hierarchical Industrial SEM Defect ClassificationFigure 1: Hierarchical Multi-Head Architecture for Intelligent Triage and Classification.🚀 Project OverviewSEM-HiNet is a deep learning framework designed for high-precision, high-speed semiconductor defect classification. It utilizes a Multi-Head Hierarchical Architecture built on an NPU-optimized MobileNetV3 backbone.The project addresses the "needle in a haystack" problem in industrial Scanning Electron Microscope (SEM) imaging by combining intelligent noise filtering (Gatekeeping) with specialized defect expertise (Specialist Heads).🧠 Technical ApproachOur approach utilizes a single shared backbone to extract high-dimensional features, which are then processed by five parallel classification heads.H1 (The Gatekeeper): Separates valid SEM data (Normal/Defect) from industrial noise and outliers (Bogus).H2 (The Router): Directs defects into three primary geometric families: Line Defects, Area Defects, or Edge Defects.H3–H5 (The Specialists): Binary experts trained to distinguish between subtle textures (e.g., Bridge vs. Open, CMP vs. Crack, and LER vs. Incomplete Etch).Key Training InnovationsStratified Partitioning: Ensures proportional class distribution across training and validation sets for stable metrics.Weighted Head Loss: Prioritizes "Expert" heads (H3-H5) during training to resolve leaf-node confusion.QAT (Quantization-Aware Training): Simulates 8-bit precision during training to ensure zero accuracy loss when deployed on the i.MX 8M Plus NPU.📊 PerformanceHeadMetricScoreH1Gatekeeper Recall~99%H2Router Accuracy100%H3-H5Specialist Precision>95%📂 Repository Structure.
├── data/                   # Dataset storage (local copies)
│   ├── normal/             # Valid SEM scans with no defects
│   ├── bogus/              # Non-SEM images (noise/outliers)
│   ├── bridge/             # Line defect: Bridge category
│   ├── open/               # Line defect: Open category
│   ├── cmp/                # Area defect: CMP category
│   ├── crack/              # Area defect: Crack category
│   ├── ler/                # Edge defect: Line Edge Roughness
│   └── inc_etch/           # Edge defect: Incomplete Etch
├── models/                 # Model architecture and weights
│   ├── arch.py             # SEMHierarchicalNetV3_QAT class definition
│   └── sem_final.pth       # Trained QAT model weights (FP32/Int8 simulated)
├── scripts/                # Utility and training scripts
│   ├── train.py            # Main stratified training & QAT loop
│   ├── evaluate.py         # Confusion matrix and recall reporting
│   └── export_onnx.py      # Conversion script for i.MX 8M Plus NPU
├── README.md               # Project documentation
└── requirements.txt        # Python dependency list
File Explanationsdata/: Organized by folder name to support the Auto-Discovery dataset class. The folder names map directly to the hierarchical label vectors.arch.py: Contains the core PyTorch model. It includes the QuantStub for the backbone and the DeQuantStub before the heads to maintain specialist accuracy.train.py: Implements the Weighted Random Sampler and Weighted Head Loss. This is where the "Specialist-Priority" training happens.evaluate.py: Generates the 5-head confusion matrices. It includes logic to handle missing classes in small validation splits.export_onnx.py: Converts the trained .pth file into an ONNX format with 8-bit quantization metadata required by the NXP NPU.🛠️ Installation & Usage# Clone the repository
git clone [https://github.com/your-username/SEM-HiNet.git](https://github.com/your-username/SEM-HiNet.git)

# Install dependencies
pip install -r requirements.txt
Running Inferencefrom models.arch import SEMHierarchicalNetV3_QAT
import torch

# Load weights and run on your SEM image
model = SEMHierarchicalNetV3_QAT()
model.load_state_dict(torch.load('models/sem_final.pth'))
model.eval()
🎯 Target HardwareThis model is optimized for:i.MX 8M Plus NPU (Primary Target)Cora Z7 (Zynq-7000)Edge devices requiring <50ms latency.
