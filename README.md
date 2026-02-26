# 🌾 Orbital Agronomy - Pre-Visual Crop Stress Detection

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI-powered early crop stress detection using hyperspectral satellite/drone imagery**

Orbital Agronomy detects plant stress **7-14 days before visible symptoms** appear, enabling farmers to take preventive action and save crops. The system analyzes spectral signatures from satellite or drone imagery to identify water stress, nutrient deficiency, and disease at the molecular level.

---

## 📊 Key Features

- **Multi-Model Architecture**: Choose from U-Net (CNN 2D), Deep MLP, or Random Forest based on your data and use case
- **Pre-Visual Detection**: Identifies stress before human-visible symptoms using spectral indices (NDWI, PRI)
- **Flexible Input**: Supports RGB+NIR imagery (drones) or 126-band hyperspectral TIF files (satellites)
- **Real-Time Analysis**: FastAPI backend with React frontend for instant stress mapping
- **Actionable Insights**: 3-tier classification (Healthy, Moderate, Severe) with specific recommendations
- **Heat Map Visualization**: Color-coded overlays show stress distribution across fields

---

## 🎯 Use Cases

| Scenario | Model | Input | Output |
|----------|-------|-------|--------|
| **Quick Field Survey** | U-Net (CNN 2D) | Drone RGB+NIR images | Spatial stress patterns, disease spread zones |
| **Precision Agriculture** | Deep MLP | Satellite hyperspectral TIF | Multi-stress classification (disease vs drought vs nutrients) |
| **Real-Time Monitoring** | Random Forest | Hyperspectral TIF | Fast baseline predictions for daily monitoring |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    React + Vite Frontend                     │
│  • Model Selector (U-Net / Deep MLP / Random Forest)         │
│  • Conditional File Upload (RGB+NIR or TIF)                  │
│  • Stress Heat Map Visualization + Charts                    │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP POST /api/analyze
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Port 8000)              │
│  • Multi-Model Routing Logic                                 │
│  • TIF Processing (rasterio)                                 │
│  • Feature Engineering (NDWI, PRI)                            │
│  • Stress Probability Calculation                            │
│  • Heat Map Overlay Generation                               │
└────────┬──────────────┬──────────────┬────────────────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
    │ U-Net   │   │ Deep MLP  │  │ Random  │
    │ PyTorch │   │ Scikit-   │  │ Forest  │
    │ (GPU)   │   │ Learn     │  │ Scikit  │
    └─────────┘   └───────────┘  └─────────┘
       4 CH           128 Features   126 Bands
    RGB+NIR         + NDWI + PRI    Spectral
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.14+ (with pip)
- Node.js 18+ (with npm)
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/smitpatil06/ENIGMA_2.0_CodeKillers.git
cd ENIGMA_2.0_CodeKillers/orbital_agronomy
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd ../frontend
npm install
cd ..
```

### Running the Application

**Option 1: Using the convenience script** (Linux/Mac)
```bash
chmod +x run_orbital_agronomy.sh
./run_orbital_agronomy.sh
```

**Option 2: Manual startup**

Terminal 1 - Backend:
```bash
cd orbital_agronomy
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

**Access the application**: Open http://localhost:5173 in your browser

---

## 📂 Project Structure

```
ENIGMA/
├── orbital_agronomy/
│   ├── app/
│   │   └── main.py                    # FastAPI backend (multi-model inference)
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── train/                 # Training data (Health/Rust/Other)
│   │   │   │   ├── Health/*.tif
│   │   │   │   ├── Rust/*.tif
│   │   │   │   └── Other/*.tif
│   │   │   ├── val/                   # Validation hyperspectral TIF files
│   │   │   └── master_training_data.csv  # Preprocessed pixel-level features
│   │   └── processed/
│   ├── models/
│   │   ├── CNN_2D/
│   │   │   └── stress_vision_weights.pth  # U-Net trained weights
│   │   └── CNN+scikit/
│   │       ├── deep_stress_model.pkl      # MLP Deep Learning model
│   │       ├── stress_vision_model.pkl    # Random Forest model
│   │       ├── label_encoder.pkl          # Class label encoder
│   │       └── data_scaler.pkl            # StandardScaler for features
│   ├── src/
│   │   ├── CNN_2D/
│   │   │   └── train_cnn_2d.py           # U-Net training script
│   │   └── Scikitlearn/
│   │       ├── image_to_tabular.py       # TIF → CSV conversion
│   │       ├── train_model.py            # Random Forest training
│   │       ├── train_deep_model.py       # Deep MLP training
│   │       ├── visualize_stress.py       # RF inference visualization
│   │       └── visualize_deep_stress.py  # MLP inference visualization
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── App.jsx                       # React frontend UI
│   ├── package.json
│   └── vite.config.js
└── run_orbital_agronomy.sh              # Convenience startup script
```

---

## 🧠 Model Comparison

### U-Net (CNN 2D) - Spatial Segmentation

**Architecture**: ResNet34 encoder + U-Net decoder with skip connections

**Input**: 4-channel tensor (RGB + NIR) at 512×512px

**Strengths**:
- ✅ Preserves spatial context (maps disease spread patterns)
- ✅ Works with common drone imagery (RGB+NIR)
- ✅ Test-Time Augmentation (TTA) for smoother predictions

**Use Cases**: Drone-based field surveys, irrigation planning, disease spread tracking

**Training**: Segmentation loss on labeled stress regions

---

### Deep MLP (CNN 1D + Scikit) - Spectral Classification

**Architecture**: 3-layer MLP (128 → 64 → 32 neurons) with ReLU activation

**Input**: 128 features per pixel (126 spectral bands + NDWI + PRI)

**Feature Engineering**:
- **NDWI** (Normalized Difference Water Index): `(NIR - SWIR) / (NIR + SWIR)`
  - Detects water stress/dehydration
  - Uses Band_90 (NIR) and Band_110 (SWIR)
  
- **PRI** (Photosynthetic Reflectance Index): `(B531 - B570) / (B531 + B570)`
  - Detects photosynthesis efficiency
  - Indicates nutrient deficiency or disease

**Strengths**:
- ✅ Captures chemical signatures (126 hyperspectral bands)
- ✅ Scientific indices validated by agronomic research
- ✅ Multi-stress classification (disease vs drought vs nutrients)

**Use Cases**: Satellite-based monitoring, precision agriculture, research

**Training**: Adam optimizer, StandardScaler normalization, 30 epochs

---

### Random Forest - Ensemble Baseline

**Architecture**: 100 decision trees with majority voting

**Input**: 126 spectral bands per pixel

**Strengths**:
- ✅ Fast training and inference (CPU-optimized)
- ✅ Robust to outliers (ensemble averaging)
- ✅ Interpretable (feature importance, decision rules)
- ✅ No feature engineering required

**Use Cases**: Quick baseline, real-time monitoring, edge deployment

**Training**: Balanced sampling (50k pixels per class), all CPU cores

---

## 📊 Performance Metrics

| Model | Accuracy | Inference Speed | Data Requirement | Interpretability |
|-------|----------|-----------------|------------------|------------------|
| **U-Net** | ~85% mIOU | 2-3 sec (GPU) | RGB+NIR images | Low (black box) |
| **Deep MLP** | ~80-85% | 1-2 sec (CPU) | Hyperspectral TIF + indices | Medium |
| **Random Forest** | ~85-90% | <1 sec (CPU) | Hyperspectral TIF only | High (feature importance) |

*Evaluated on custom hyperspectral dataset (2.3M pixel samples)*

---

## 🔬 Scientific Background

### Remote Sensing Physics

**Healthy Plants**:
- High NIR reflectance (cellular structure intact)
- Low red absorption (active chlorophyll photosynthesis)

**Stressed Plants**:
- Low NIR reflectance (cell damage)
- High red reflectance (chlorophyll breakdown)

**Hyperspectral Advantage**:
- 126 bands capture absorption peaks for:
  - **Water**: Band_110 (SWIR, 1550nm)
  - **Nitrogen**: Band_75 (Red Edge, 720nm)
  - **Chlorophyll**: Band_45 (Green Peak, 531nm)

### Feature Engineering Rationale

**NDWI (Water Stress)**:
- Based on NASA research
- Water absorbs SWIR (1550nm) but reflects NIR (900nm)
- Ratio drops when plants dehydrate

**PRI (Photosynthetic Stress)**:
- Based on plant physiology
- Xanthophyll cycle changes reflectance at 531nm under stress
- 570nm is reference wavelength

---

## 🎨 UI Features

### Upload Telemetry
- **U-Net Mode**: Upload RGB + NIR images (JPG/PNG)
- **Deep/RF Mode**: Upload hyperspectral TIF files

### Model Selector
- 3-button toggle: U-Net | Deep MLP | Random Forest
- Descriptions and use-case hints

### Stress Map Output
- **Raw Preview**: Original RGB composite
- **Stress Overlay**: Color-coded heat map
  - 🟢 Green: Healthy (prob < 0.35)
  - 🟡 Yellow: Moderate stress (0.35 - 0.70)
  - 🔴 Red: Severe stress (prob > 0.70)

### Analytics Dashboard
- **Health Breakdown**: Donut chart showing distribution
- **Pixel Severity Count**: Bar chart with absolute counts
- **System Status**: Actionable recommendations based on results

---

## 📖 Training Your Own Models

### 1. Prepare Dataset

Organize hyperspectral TIF files:
```
data/datasets/train/
├── Health/
│   ├── 001.tif
│   ├── 002.tif
│   └── ...
├── Rust/
│   ├── 101.tif
│   ├── 102.tif
│   └── ...
└── Other/
    ├── 201.tif
    └── ...
```

### 2. Convert Images to Tabular

```bash
python orbital_agronomy/src/Scikitlearn/image_to_tabular.py
```

This extracts pixel-level spectral signatures and creates `master_training_data.csv` with 2.3M samples.

### 3. Train Models

**Random Forest**:
```bash
python orbital_agronomy/src/Scikitlearn/train_model.py
```

**Deep MLP**:
```bash
python orbital_agronomy/src/Scikitlearn/train_deep_model.py
```

**U-Net CNN 2D**:
```bash
python orbital_agronomy/src/CNN_2D/train_cnn_2d.py
```

Models automatically save to `models/` directory.

### 4. Test Inference

**Random Forest**:
```bash
python orbital_agronomy/src/Scikitlearn/visualize_stress.py
```

**Deep MLP**:
```bash
python orbital_agronomy/src/Scikitlearn/visualize_deep_stress.py
```

---

## 🛠️ API Reference

### POST `/api/analyze`

**Request** (multipart/form-data):

For U-Net:
```
rgb_file: <file>      # RGB image (JPG/PNG)
nir_file: <file>      # NIR image (JPG/PNG)
model_type: "unet"
```

For Deep MLP / Random Forest:
```
tif_file: <file>      # Hyperspectral TIF (126 bands)
model_type: "deep"    # or "rf"
```

**Response** (JSON):
```json
{
  "status": "success",
  "image_base64": "data:image/jpeg;base64,...",
  "preview_base64": "data:image/jpeg;base64,...",
  "model_type": "Random Forest Classifier",
  "metrics": {
    "healthy_percent": 81.1,
    "moderate_percent": 9.9,
    "severe_percent": 8.9,
    "stressed_percent": 18.8,
    "severity": {
      "healthy": 523401,
      "moderate": 63854,
      "severe": 57392
    },
    "mean_stress": 0.2347
  }
}
```

### GET `/api/models/status`

**Response**:
```json
{
  "unet": true,
  "deep_learning": true,
  "random_forest": true
}
```

---

## 🔧 Configuration

### Backend (FastAPI)

Edit `orbital_agronomy/app/main.py`:

```python
# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Stress thresholds (adjust sensitivity)
healthy  = stress_map < 0.35   # Lower = more sensitive
moderate = stress_map >= 0.35 and stress_map < 0.70
severe   = stress_map >= 0.70   # Higher = less false positives
```

### Frontend (React)

Edit `frontend/src/App.jsx`:

```javascript
// API endpoint
const API_URL = 'http://localhost:8000/api/analyze';

// Color scheme
const COLORS = {
  healthy: '#22c55e',   // Green
  moderate: '#eab308',  // Yellow
  severe: '#ef4444'     // Red
};
```

---

## 🐛 Troubleshooting

### Backend Issues

**Model not loading**:
```bash
# Check if model files exist
ls orbital_agronomy/models/CNN_2D/
ls orbital_agronomy/models/CNN+scikit/

# Retrain if missing
python orbital_agronomy/src/Scikitlearn/train_model.py
```

**CUDA out of memory**:
```python
# In app/main.py, force CPU mode
device = torch.device("cpu")
```

**TIF reading errors**:
```bash
# Install rasterio with GDAL
pip install rasterio --no-binary rasterio
```

### Frontend Issues

**CORS errors**:
```python
# In app/main.py, add your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add specific origin
)
```

**Charts not rendering**:
```bash
# Reinstall Recharts
cd frontend
npm install recharts --force
```

---

## 📚 Dataset Information

**Source**: Custom hyperspectral imagery dataset

**Specifications**:
- 126 spectral bands (400nm - 2500nm)
- Spatial resolution: Variable (0.5m - 10m per pixel)
- Classes: Health, Rust (disease), Other (mixed stress)
- Total samples: 2.3 million pixel-level spectral signatures

**Training Split**:
- Balanced sampling: 40k-50k pixels per class
- Train/Test: 80/20 split
- Validation: Separate TIF files in `data/datasets/val/`

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Commit with conventional commits: `git commit -m 'feat: add time-series analysis'`
4. Push to branch: `git push origin feat/amazing-feature`
5. Open a Pull Request

### Commit Convention

- `feat` - New feature added
- `fix` - Bug fixed
- `docs` - Documentation changes
- `style` - Formatting only (no logic change)
- `refactor` - Code improved, same behavior
- `perf` - Performance improved
- `test` - Tests added/updated
- `chore` - Maintenance work (deps, configs)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Team CodeKillers** - ENIGMA 2.0 Hackathon

- GitHub: [@smitpatil06](https://github.com/smitpatil06)
- Repository: [ENIGMA_2.0_CodeKillers](https://github.com/smitpatil06/ENIGMA_2.0_CodeKillers)

---

## 🙏 Acknowledgments

- **Agriculture-Vision Dataset** for baseline evaluation
- **NASA Remote Sensing Research** for NDWI/PRI indices
- **segmentation-models-pytorch** for U-Net implementation
- **FastAPI** and **React** communities for excellent frameworks

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Documentation: See `/docs` folder (if available)

---

## 🔮 Future Enhancements

- [ ] Temporal analysis (LSTM for time-series predictions)
- [ ] Mobile app for field technicians
- [ ] Cloud deployment (AWS/GCP)
- [ ] Multi-crop support (corn, wheat, rice)
- [ ] Integration with weather APIs
- [ ] Prescription map generation (variable rate application)
- [ ] Blockchain-based crop insurance validation

---

**Built with ❤️ for sustainable agriculture**

*Making precision farming accessible to everyone*
