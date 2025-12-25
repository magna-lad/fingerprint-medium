# Hybrid Fingerprint Recognition System

This project implements a multi-modal biometric authentication system that fuses **Geometric Graph Analysis** (using Minutiae topology) with **Deep Texture Analysis** (using CNNs with Spatial Transformer Networks).

## 📂 Project Structure

The project is divided into processing, feature extraction, and model training modules.

### 1. Core Execution
*   **`main.py`**
    *   **Purpose:** The entry point for data preprocessing.
    *   **Function:** Reads raw images, applies Gabor filtering, extracts skeletons/minutiae, and caches the results to `processed_data.pkl`.
*   **`run_final_pipeline.py`**
    *   **Purpose:** The main training and evaluation script.
    *   **Function:** Loads the cached data, creates graph pairs, trains the XGBoost model, trains the CNN ensemble, and calculates the final fusion score.

### 2. Preprocessing & Minutiae Extraction
*   **`reader.py`**: Iterates through the specific directory structure (`User/Hand/Finger`) to load images.
*   **`minutia_loader.py`**: Handles ROI segmentation, normalization, and masking.
*   **`skeleton_maker.py`**:
    *   Computes ridge orientation fields.
    *   Estimates ridge frequency.
    *   Applies Gabor filters.
    *   Performs skeletonization (thinning).
*   **`minutiaeExtractor.py`**: Uses the Crossing Number method on the skeleton to find ridge endings and bifurcations.
*   **`minutiae_filter.py`**: filtering logic to remove false minutiae near image borders or in high-noise areas.

### 3. Geometric Branch (Graph + XGBoost)
*   **`graph_minutiae.py`**:
    *   Converts minutiae points into a graph using Delaunay Triangulation.
    *   Handles dataset splitting (Train/Val/Test) disjoint by user.
*   **`xgboost_feature_extractor.py`**:
    *   Extracts handcrafted features from graph pairs (Edge length histograms, angle histograms, spatial density).

### 4. Visual Branch (Deep Learning)
*   **`cnn_model.py`**:
    *   **STN (Spatial Transformer Network):** Aligns fingerprint patches automatically.
    *   **DeeperCNN:** A ResNet-based Siamese network for embedding extraction.
    *   **FingerprintTextureDataset:** Handles core-alignment and data augmentation.

### 5. Utilities
*   **`load_save.py`**: Manages the loading and saving of the `processed_data.pkl` file in the `biometric_cache` directory.

---

## ⚙️ Installation

### Prerequisites
*   Python 3.8+
*   CUDA-capable GPU (recommended for CNN training)

### Install Dependencies
```bash
pip install numpy opencv-python matplotlib tqdm torch torchvision torchaudio
pip install xgboost scikit-learn scipy scikit-image torch-geometric pandas
```