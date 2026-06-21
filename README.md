# MABe Challenge — Social Action Recognition in Mice

> 🏆 **22nd out of 1412 teams** | Top 2% | Kaggle Competition

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-02b152)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Competition Overview

The [MABe (Mouse Action Behavior) Challenge](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection) is a Kaggle competition focused on automatically recognizing social actions in mice from multi-animal pose estimation tracking data. Given keypoint trajectories (nose, ears, body center, tail base, etc.), the goal is to predict the start/stop frames and action types for each pair of interacting mice.

### Actions to Recognize (Multi-label Temporal Segmentation)

| Type | Actions |
|------|---------|
| **Single-mouse** | rear, groom, lick, sniff, etc. |
| **Pair (social)** | chase, flee, attack, mount, investigate, sniff face/body/genital, etc. |

<p align="center">
  <img src="RANK.png" alt="Competition Ranking" width="500"/>
</p>

## 🧠 Methodology

### Model Pipeline

```
Raw Tracking Data (parquet)
    ↓ pivot by mouse_id × bodypart
    ↓ coordinate normalization (cm)
    ↓ missing-value gap filling
    ↓
┌─ Single-mouse Features ─┐   ┌─ Pair Interaction Features ─┐
│ • Body part distances   │   │ • Cross body-part distances │
│ • Motion (vel/acc/curv) │   │ • Chase/Flee/Sidestep      │
│ • Posture & orientation │   │ • Relative trajectory      │
│ • Arena spatial context │   │ • Contact semantics        │
│ • High-freq micro-motion│   │ • Leader-follower dynamics │
│ • Body-axis decomposition│  │ • Body overlap detection   │
│ • Missingness-as-signal │   │ • Speed/angle similarity   │
└─────────────────────────┘   └────────────────────────────┘
    ↓
Meta features (fps, arena shape, n_mice, lab_id...)
    ↓
LightGBM Binary Classifier × each action
    ↓ StratifiedGroupKFold (5-fold, by video_id)
    ↓ Optuna threshold tuning per action
    ↓ predict_multiclass → merge consecutive frames → submission
```

### Key Techniques

- **FPS-adaptive windows**: All temporal windows scale with actual video frame rate
- **Missingness as signal**: Keypoint occlusion patterns inform aggressive/social behaviors
- **Proxy body parts**: Graceful fallback when certain keypoints are unavailable (e.g., nose→head→ear midpoint)
- **Body-axis decomposition**: Decompose velocity into forward/lateral relative to body orientation
- **Contact semantics**: Multi-threshold front-anchor-to-body distances detect physical interactions
- **Circular & rectangular arenas**: Arena geometry-aware spatial features

## 📂 Project Structure

```
MABe-Challenge---Social-Action-Recognition-in-Mice/
├── MabeTrain.py          # Main training & inference pipeline (~3200 lines)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── RANK.png              # Competition ranking screenshot
└── .gitignore
```

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit the `CFG` class in `MabeTrain.py`:

```python
class CFG:
    # Data paths (Kaggle environment)
    train_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    test_path = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    train_annotation_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    test_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    # Run mode
    mode = "validate"   # "validate" for CV evaluation, "submit" for test prediction
    n_splits = 5
```

### 3. Run Training / Validation

```bash
python MabeTrain.py
```

- **`mode = "validate"`**: Performs 5-fold cross-validation on the training set, outputs per-action F1 scores and saves models/thresholds.
- **`mode = "submit"`**: Loads pre-trained models and generates `submission.csv` for the test set.

### 4. Kaggle Notebook

This code is designed to run in the Kaggle environment. To reproduce:
1. Create a Kaggle Notebook with GPU enabled
2. Add the [MABe competition dataset](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data)
3. Copy `MabeTrain.py` and run

## 📊 Results

| Metric | Score |
|--------|-------|
| **Competition Metric** | Custom F1-based multi-label segmentation score |
| **Ranking** | 22 / 1412 (Top 1.6%) |
| **Mean Binary F1 (CV)** | ~0.XX (varies by section/action) |

## 🛠 Dependencies

- **Python 3.8+**
- **LightGBM** — Gradient boosting classifier
- **scikit-learn** — Cross-validation, metrics
- **Optuna** — Hyperparameter / threshold optimization
- **pandas / polars** — Data manipulation
- **numpy** — Numerical computing
- **joblib** — Model serialization & parallel processing
- **koolbox** — Custom training utilities (Kaggle-specific)
- **tqdm** — Progress bars

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [MABe Challenge](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection) organizers and dataset providers
- Kaggle community for invaluable discussions and insights
- All participating teams for pushing the boundaries of animal behavior recognition
