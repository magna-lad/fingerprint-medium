# fingerprint-medium

End-to-end minutiae-based fingerprint verification pipeline built from classical image processing, graph construction, and gradient-boosted decision trees (XGBoost / LightGBM).  
The code takes raw fingerprint images, extracts minutiae, converts them into graphs, and trains a verifier that scores whether two impressions come from the same finger.

---

## 1. Project goals

- Implement a **fully classical** fingerprint pipeline (no CNNs) with:
  - ROI segmentation, normalization, orientation & ridge-frequency estimation, Gabor enhancement, and skeletonization.
  - Minutiae extraction (ridge endings / bifurcations) and post-filtering.
- Represent each fingerprint as a **graph of minutiae** with geometry- and orientation-aware node/edge features.
- Train a **pairwise verifier** using XGBoost (or LightGBM) on engineered graph features, with:
  - Genuine vs impostor pairs.
  - User-disjoint train / test splits.
  - Data augmentation at the minutiae-graph level.

---

## 2. Repository structure

fingerprint-medium/
├─ main.py
├─ run_xgboost_pipeline.py
├─ graph_minutiae.py
├─ xgboost_feature_extractor.py
├─ minutia_loader.py
├─ skeleton_maker.py
├─ minutiaeExtractor.py
├─ minutiae_filter.py
├─ reader.py
├─ load_save.py
└─ .gitignore



### Core scripts

- **`main.py`**  
  - Entry point for **preprocessing** raw fingerprint images.  
  - Loads users and impressions, runs:
    - Normalization & ROI segmentation (`minutia_loader.py`).
    - Orientation, ridge-frequency, Gabor filtering, skeletonization, and minutiae extraction (`skeleton_maker.py` + `minutiaeExtractor.py`).
    - Minutiae filtering / cleaning (`minutiae_filter.py`).  
  - Saves a processed dictionary to `biometric_cache/processed_data.pkl`.

- **`run_xgboost_pipeline.py`**  
  - Entry point for the **ML pipeline**:
    - Loads `processed_data.pkl` via `load_save.load_users_dictionary`.
    - Builds one graph per impression using `GraphMinutiae` (`graph_minutiae.py`).
    - Creates genuine & impostor pairs and splits them into **user-disjoint** train / test sets.
    - Extracts pairwise feature vectors using `xgboost_feature_extractor.py`.
    - Trains an XGBoost classifier with `RandomizedSearchCV` hyperparameter tuning.
    - Computes ROC, AUC, EER, basic accuracy, and prints simple error analysis.
    - Saves the best model as `fingerprint_verifier.joblib`.

### Preprocessing modules

- **`reader.py`**  
  - Expects a root dataset directory like:
    ```
    data_dir/
      ├─ 000/
      │   ├─ L/
      │   │   ├─ 000_L0_0.bmp ... 000_L3_4.bmp
      │   └─ R/
      │       ├─ 000_R0_0.bmp ... 000_R3_4.bmp
      ├─ 001/
      └─ ...
    ```
  - Builds a nested structure:
    ```
    users = {
      "000": {
        "fingers": {
          "L": [[impr1, ..., impr5],  # finger 0
                ...],                 # fingers 1–3
          "R": [[...], ...]
        }
      },
      ...
    }
    ```
    where each `impr` is a grayscale image (`numpy` array).

- **`minutia_loader.py`** (`minutiaLoader` class)  
  - Takes a grayscale fingerprint image and:
    - Normalizes it by global mean/std.
    - Segments ROI block-wise using local variance with a global threshold.
    - Smooths the ROI mask using morphological open/close operations.
    - Outputs `(segmented_img, norm_img, mask)` plus a normalized image for later stages.

- **`skeleton_maker.py`** (`skeleton_maker` class)  
  - Consumes normalized + segmented images and mask:
    - Computes **block-wise orientation field** using Sobel gradients (double-angle trick).
    - Optionally smooths orientations.
    - Estimates **ridge frequency** per block and computes a global median frequency map.
    - Applies **Gabor filtering** with tuned parameters from orientation/frequency.
    - Produces a **skeletonized** binary image (via `skimage.morphology.skeletonize`).
    - Stores orientation map (`angle_gabor`) and frequency (`freq`) for downstream use.

- **`minutiaeExtractor.py`** (`minutiaeExtractor` class)  
  - Operates on skeleton images:
    - Uses **crossing number** on 8-neighborhood to classify:
      - CN = 1 → ridge ending.
      - CN = 3 → bifurcation.
    - Estimates minutiae orientation in a local window via PCA of ridge pixels.
    - Removes minutiae too close to the image borders.
    - Returns a list of `(x, y, angle, type)` tuples.

- **`minutiae_filter.py`** (`minutiae_filter` class)  
  - Cleans up raw minutiae:
    - Binarizes ROI mask and builds a **boundary band** around both mask and image edges.
    - Removes minutiae on or near the combined boundary.
    - Enforces a **minimum distance** between minutiae.
    - Checks **local orientation consistency** (within a radius) and discards inconsistent points.
  - Returns a filtered skeleton image and filtered minutiae list.

- **`load_save.py`**  
  - `save_users_dictionary(users, filename)`:
    - Saves processed `users` dict with `pickle` into `biometric_cache/<filename>`.
  - `load_users_dictionary(filename, calc_roc=False)`:
    - Loads processed dictionary if it exists.
    - Optionally prints basic stats about the cache.

At the end of `main.py`, each impression entry becomes a dictionary such as:

{
"skeleton": np.ndarray, # 2D skeleton image
"minutiae": np.ndarray, # shape (N, 4): x, y, angle, type
"mask": np.ndarray, # ROI mask
"orientation_map": np.ndarray # block-wise orientation field
}


---

## 3. Graph construction and features

- **`graph_minutiae.py`** (`GraphMinutiae` class)

  - **Node features** (per minutia):
    - Coordinates centered on an estimated core point (x', y').
    - Minutiae type one-hot (ridge ending / bifurcation).
    - Orientation encoded as `(sin θ, cos θ)`.
    - Distance to core.
    - Direction from minutia to core, encoded as `(sin φ, cos φ)`.
    - Raw minutiae type appended as a numeric feature.

  - **Core estimation**:
    - Primary method: `find_true_core(orientation_map)` using **Poincaré index** on the block-wise orientation field to locate singularities (cores / deltas).
    - Fallback: `find_core_proxy(minutiae)` using the centroid of minutiae when no core is detected.

  - **Edges**:
    - Built via **Delaunay triangulation** on normalized minutiae coordinates.
    - Stored as an undirected graph (edges duplicated in both directions).
    - Edge attributes:
      - Euclidean distance between nodes.
      - Relative angle between local orientations of incident minutiae.
      - Orientation of the edge vector itself.

  - **Graph creation**:
    - `graph_maker()`:
      - Iterates over all users / hands / fingers / impressions in the processed dictionary.
      - Builds one graph per impression (skipping low-minutiae cases).
      - Saves:
        - `fingerprint_graphs`: list of graph objects.
        - `graph_metadata`: mapping from `graph_id` to user/hand/finger/impression indices.

  - **Pair generation and splitting**:
    - `create_graph_pairs(num_impostors_per_genuine=...)`:
      - **Genuine pairs**: different impressions of the same user, same hand, same finger.
      - **Impostor pairs**: random pairs of graphs from different users, up to a desired ratio.  
    - `get_user_splits(train_ratio, val_ratio)`:
      - Splits **users** into train/val/test sets (disjoint IDs).  
    - `split_pairs_by_user(all_pairs, train_users, val_users, test_users)`:
      - Ensures each pair is entirely within one split, preventing user leakage.

- **`xgboost_feature_extractor.py`**

  - `get_graph_summary_features(graph)`:
    - Aggregates a **single fixed-length feature vector per graph**:
      - Global stats: #nodes, #edges, average degree.
      - Node-feature stats: mean and std of selected attributes.
      - Distance-to-core stats: median, 25th, 75th percentile.
      - Edge-distance stats: mean, std, median, quartiles + histogram over distance bins.
      - Edge-angle stats: mean, std, median.
      - Minutiae-type stats: count of endings / bifurcations and their ratio.
      - Radial profiling: angular statistics in concentric rings around the core.
      - Local neighborhood distances (k-NN based statistics).
      - Local **relative orientation** features between a node and its neighbors.

  - `create_feature_vector_for_pair(graph1, graph2)`:
    - Computes features for each graph separately.
    - Concatenates:
      - Absolute difference of features.
      - Element-wise product of features.  
    - Output: a symmetric, order-invariant feature vector describing a pair.

---

## 4. Model training pipeline

The end-to-end training script is `run_xgboost_pipeline.py`.

### 4.1 Data preparation

- Loads processed users:
users = load_users_dictionary('processed_data.pkl', True)

- Builds all graphs with `GraphMinutiae(users).graph_maker()`.
- Constructs genuine + impostor pairs via `create_graph_pairs(num_impostors_per_genuine=4)`.
- Splits pairs into train / test based on user IDs:
train_users, _, test_users = analyzer.get_user_splits()
train_pairs, _, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, [], test_users)



- `prepare_data_with_augmentation(pairs, is_training=False, aug_multiplier=3)`:
- For each pair `(g1, g2, label)`:
  - Computes a pairwise feature vector.
  - If `is_training` and `label == 1` (genuine), creates additional augmented pairs:
    - Augments raw minutiae of `g1` with random rotation + translation.
    - Rebuilds a new graph on-the-fly and recomputes features.
- Returns `(X, y)` as `numpy` arrays.

### 4.2 Model and hyperparameter search

- Base model: `xgboost.XGBClassifier` with:
- `objective='binary:logistic'`
- `eval_metric='logloss'`
- `scale_pos_weight` computed from class imbalance.
- Hyperparameter search with `RandomizedSearchCV`:
- Samples over:
  - `n_estimators`, `max_depth`, `learning_rate`
  - `subsample`, `colsample_bytree`
  - `gamma`, `reg_lambda`
- Optimizes **ROC-AUC** with 3-fold CV.
- Uses all CPU cores (`n_jobs=-1`).

The best model is saved as:

fingerprint_verifier.joblib


### 4.3 Evaluation

`evaluate_xgboost_model(...)` computes:

- Accuracy at a 0.5 threshold.
- ROC curve + AUC.
- Equal Error Rate (EER) and corresponding score threshold.
- Plots:
  - ROC curve with EER point highlighted.
  - Feature importance (gain-based) for the top features.
- Simple error analysis:
  - Lists worst **false negatives** (missed genuine pairs).
  - Lists worst **false positives** (impostors accepted), including graph IDs and scores.

---

## 5. Installation & setup

### 5.1 Environment

Recommended:

- Python 3.9+ (3.10 also typically works).
- A recent `pip` and virtual environment (`venv` or `conda`).

Install core dependencies (you may already have some):

pip install numpy scipy scikit-image scikit-learn matplotlib tqdm opencv-python joblib xgboost lightgbm


For graph processing:

- `torch` (CPU or GPU build).  
- `torch-geometric` (install per the official instructions, as it depends on your CUDA / PyTorch version).

### 5.2 Dataset

1. Download / prepare your fingerprint dataset (e.g., a Kaggle fingerprint database).
2. Organize it as:

```
<DATA_DIR>/
├─ 000/
│ ├─ L/
│ │ ├─ 000_L0_0.bmp ... 000_L3_4.bmp
│ └─ R/
│ ├─ 000_R0_0.bmp ... 000_R3_4.bmp
├─ 001/
└─ ...
```

3. In `main.py`, set `data_dir` to your `<DATA_DIR>` path.

---

## 6. Running the pipeline

### 6.1 Step 1 – Preprocess and cache

python main.py


- If `biometric_cache/processed_data.pkl` exists, it will be loaded and reused.
- Otherwise, the script:
  - Loads raw images.
  - Runs segmentation, enhancement, skeletonization, minutiae extraction, and filtering.
  - Saves the resulting user dictionary to `biometric_cache/processed_data.pkl`.

### 6.2 Step 2 – Train & evaluate the verifier

python run_xgboost_pipeline.py


This will:

- Build minutiae graphs for all impressions.
- Create genuine and impostor pairs with user-disjoint train/test splits.
- Extract pairwise features with augmentation on genuine training pairs.
- Run hyperparameter search and train the best XGBoost model.
- Print metrics (AUC, EER, accuracy) and display ROC + feature-importance plots.
- Save the trained model as `fingerprint_verifier.joblib`.

---

