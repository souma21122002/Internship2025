# Deepfake Detection Using Hand-Crafted Features and Deep Learning

A multi-modal deepfake detection pipeline combining classical forensic features (ELA, SRM, DCT) with dual-stream Convolutional Neural Networks, achieving up to **99.8% accuracy** and **~1.0 AUC** across multiple benchmark datasets.

> Summer Research Internship (May–July 2025)
> Indian Institute of Engineering Science and Technology, Shibpur (IIEST Shibpur)
> Department of Information Technology

---

## 👤 Author

**Soumadip Singha Mahapatra** (Roll No. 2022ITB097)
B.Tech, Information Technology, IIEST Shibpur

Teammate: **Ankesh Hatui**

Under the guidance of **Dr. Ruchira Naskar**, Associate Professor, Dept. of Information Technology, IIEST Shibpur

---

## 📌 Overview

Deepfake technology — powered by GANs, autoencoders, and face-reenactment methods like Face2Face and Neural Textures — can generate highly realistic synthetic media, posing risks to political integrity, identity security, financial systems, and legal evidence. This project builds a robust, multi-modal detection system that fuses **classical forensic signal analysis** with **deep learning** to reliably distinguish real faces from manipulated ones.

### Objectives
- Process multiple real/fake video datasets end-to-end
- Detect and extract faces efficiently from video frames
- Apply complementary forensic feature extraction techniques
- Train and compare single-stream vs. dual-stream CNN architectures
- Achieve high, balanced accuracy (precision–recall) with strong generalization
- Validate performance via k-fold cross-validation across 7 datasets

---

## 🧩 Pipeline

```
Videos (Real/Fake)
   → Frame Extraction
   → Face Extraction (MediaPipe)
   → Forensic Feature Extraction (ELA / SRM / DCT)
   → Train/Val/Test Split (80/10/10) + Path Obfuscation
   → Single-Stream / Dual-Stream CNN
   → Prediction (Real / Fake)
```

---

## 🔍 Hand-Crafted Forensic Features

### 1. Error Level Analysis (ELA)
Detects manipulation by re-compressing an image (JPEG quality 85) and computing the pixel-wise difference from the original. Uneven compression artifacts expose spliced, retouched, or recompressed regions.

### 2. Spatial Rich Model (SRM) Noise Residuals
Applies a fixed bank of **30 high-pass filters** across all 3 color channels (90 filtered responses total), then averages and normalizes the result. Captures high-frequency artifacts, GAN fingerprints, and resampling/interpolation traces invisible to the naked eye.

### 3. Discrete Cosine Transform (DCT)
Transforms each color channel into the frequency domain, averages the channels, and extracts the most informative **16×16 low-frequency block** (~90% of image energy). Reveals GAN-generated frequency anomalies and recompression inconsistencies.

---

## 🧠 Model Architectures

### Single-Stream CNN
Takes one forensic feature (ELA, SRM, or DCT) as input.

```
Input (128×128×3)
→ Conv2D(32) + BatchNorm → Conv2D(32) + BatchNorm
→ MaxPool(2×2) → Dropout(0.2)
→ Flatten → Dense(256) + BatchNorm → Dropout(0.5)
→ Dense(1, Sigmoid) → Real/Fake
```
- ~33M parameters, ~2× faster training/inference than dual-stream
- Serves as the baseline for comparison

### Dual-Stream CNN
Processes two forensic features (e.g., ELA + SRM) in parallel branches, fused via concatenation.

```
ELA Input (128×128×3)         SRM Input (128×128×3)
   → Conv2D+BN ×2                → Conv2D+BN ×2
   → MaxPool → Dropout            → MaxPool → Dropout
        \                          /
         → Concatenate (64 channels)
         → Flatten → Dense(256) + BatchNorm → Dropout(0.5)
         → Dense(1, Sigmoid) → Real/Fake
```
- ~67M parameters
- Combines complementary spatial (ELA) and frequency-noise (SRM) cues for stronger, more stable performance

---

## 📊 Datasets

| Dataset | Description |
|---|---|
| VidTIMIT & DFTIMIT | Audio-visual identity dataset + deepfake variant |
| Celeb-DF | High-quality celebrity deepfake dataset |
| FaceForensics++ (DF) | DeepFakes manipulation |
| FaceForensics++ (FS) | FaceSwap manipulation |
| FaceForensics++ (F2F) | Face2Face manipulation |
| FaceForensics++ (NT) | NeuralTextures manipulation |
| DFDC | DeepFake Detection Challenge dataset |

**Data split:** 80% train / 10% validation / 10% test, with filename obfuscation to prevent metadata leakage and custom memory-efficient Keras generators for on-demand loading.

---

## 📈 Results

Best-performing configurations per dataset (Accuracy % / AUC):

| Dataset | Best Method | Accuracy | AUC |
|---|---|---|---|
| FF-FS | SRM | 99.78 | 0.9995 |
| FF-F2F | DCT-SRM | 98.37 | 0.9988 |
| FF-NT | DCT-SRM | 97.20 | 0.9957 |
| VidTIMIT-DFTIMIT | ELA-SRM / ELA | 99.99 | 1.0000 |
| Celeb-DF | DCT-ELA | 99.89 | 0.9968 |
| FF-DF | ELA | 98.89 | 0.9966 |
| DFDC | DCT-SRM | 99.88 | 0.9999 |

Full per-dataset, per-method results (single-stream: ELA, SRM, DCT; dual-stream: ELA-SRM, DCT-ELA, DCT-SRM) are available in the [project report](#-report).

### Key Findings
- **SRM** is the strongest standalone feature, consistently achieving the highest accuracy (up to 99.87%) and AUC (~0.9998).
- **DCT** alone is the weakest feature, especially on challenging datasets like FF-NT (73.72% accuracy).
- **ELA-SRM fusion** is the strongest dual-stream combination, reaching 98–99.8% accuracy with AUC nearing 1.0, confirming that spatial and frequency-noise cues are complementary.
- Performance remains above 98% across diverse manipulation types (FaceSwap, DeepFakes, Face2Face, NeuralTextures) and real-world data (DFDC), demonstrating strong generalization.

---

## 🔮 Future Scope

1. **Triple-stream fusion** (ELA + SRM + DCT) for stronger robustness
2. **Temporal modeling** (3D-CNN, LSTM, Optical Flow) for video-level detection of motion artifacts
3. **Additional forensic cues** — chroma inconsistencies, color spatial correlation, noiseprint analysis
4. **Cross-dataset generalization & adversarial testing** on unseen/adversarially-optimized deepfakes
5. **Real-time deployment optimization** via pruning, quantization, and ONNX conversion for edge/social-media pipelines

---

## 🛠️ Tech Stack

- **Face Detection:** MediaPipe
- **Deep Learning:** TensorFlow / Keras (CNN, dual-stream architecture)
- **Forensic Feature Engineering:** OpenCV, NumPy, PIL (ELA, SRM, DCT)
- **Training:** K-fold cross-validation, Batch Normalization, Dropout regularization

---

## 📄 Report

Full methodology, literature review, complete results tables across all 7 datasets, and discussion are available in the project report: `Report_DeepfakeDetection_SoumadipSinghaMahapatra.pdf`

---

## 🙏 Acknowledgements

We thank **Dr. Ruchira Naskar** (Associate Professor, Dept. of IT, IIEST Shibpur) for her guidance and support throughout this internship, and the Department of Information Technology, IIEST Shibpur, for the academic resources and research environment.

---

## 📚 References

1. Dey, P., Chakraborty, S., & Chatterjee, K. (2024). Detection of image tampering using deep learning, error levels and noise residuals. *Neural Processing Letters, 56*, 112.
2. Agarwal, S., Farid, H., Gu, Y., He, M., Nagano, K., & Li, H. (2019). Protecting world leaders against deep fakes. *CVPRW*.
3. Yang, X., Li, Y., & Lyu, S. (2019). Exposing deep fakes using inconsistent head poses. *ICASSP*.
4. Han, B., Han, X., Zhang, H., Li, J., & Cao, X. (2021). Fighting fake news: Two-stream network for deepfake detection via learnable SRM. *IEEE T-BIOM, 3*(3), 320–333.
5. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. *ICCV*.
6. Li, Y., Chang, M., & Lyu, S. (2018). In ictu oculi: Exposing AI-generated fake face videos by detecting eye blinking. *WIFS*.
7. Thies, J., Zollhöfer, M., Stamminger, M., Theobalt, C., & Nießner, M. (2019). Face2Face: Real-time face capture and reenactment of RGB videos. *CVPR*.
8. Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: A compact facial video forgery detection network. *WIFS*.
9. Fridrich, J., & Kodovský, J. (2012). Rich models for steganalysis of digital images. *IEEE T-IFS*.
