# enhanced_roc_analysis.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import pandas as pd
from scipy import stats

class scores_roc:
    """
    Advanced ROC analysis class for fingerprint biometric systems
    Incorporates local feature analysis, preprocessing, and minutiae extraction
    """
    
    def __init__(self, users, use_minutiae=True, use_preprocessing=True):
        """
        Initialize the enhanced ROC analyzer
        
        Args:
            users: Dictionary with user_id -> {'finger': [skeleton_arrays]}
            use_minutiae: Whether to include minutiae-based similarity
            use_preprocessing: Whether to apply skeleton preprocessing
        """
        self.users = users
        self.use_minutiae = use_minutiae
        self.use_preprocessing = use_preprocessing
        
        # Results storage
        self.genuine_scores = []
        self.impostor_scores = []
        self.processed_skeletons = []
        self.user_labels = []
        self.minutiae_cache = {}
        
        print(f"Initializing enhanced ROC analysis for {len(users)} users")
        print(f"Minutiae analysis: {use_minutiae}")
        print(f"Preprocessing: {use_preprocessing}")
        
        self._extract_and_preprocess_data()
        self._calculate_enhanced_similarity_scores()
    
    def _extract_and_preprocess_data(self):
        """Extract and preprocess skeleton data"""
        print("Extracting and preprocessing skeleton data...")
        
        for user_id, finger_data in self.users.items():
            for idx, skeleton in enumerate(finger_data['finger']):
                if skeleton is not None and skeleton.size > 0:
                    # Apply preprocessing if enabled
                    if self.use_preprocessing:
                        processed_skel = self._preprocess_skeleton(skeleton)
                    else:
                        processed_skel = skeleton
                    
                    self.processed_skeletons.append(processed_skel)
                    self.user_labels.append(user_id)
                    
                    # Cache minutiae if enabled
                    if self.use_minutiae:
                        cache_key = f"{user_id}_{idx}"
                        self.minutiae_cache[cache_key] = self._extract_skeleton_minutiae(processed_skel)
        
        print(f"Processed {len(self.processed_skeletons)} skeleton templates")
    
    def _preprocess_skeleton(self, skeleton):
        """Normalize and align skeleton for consistent comparison"""
        
        # Convert to proper shape if flattened
        if len(skeleton.shape) == 1:
            size = int(np.sqrt(len(skeleton)))
            skeleton = skeleton.reshape(size, size)
        
        # 1. Normalize pixel values
        if skeleton.max() > skeleton.min():
            skeleton = (skeleton - skeleton.min()) / (skeleton.max() - skeleton.min())
        
        # 2. Center the skeleton (find center of mass)
        skeleton_binary = (skeleton > 0.1).astype(np.uint8)
        
        if np.sum(skeleton_binary) > 0:
            # Find center of mass
            moments = cv2.moments(skeleton_binary)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Translate to center
                h, w = skeleton.shape
                center_x, center_y = w // 2, h // 2
                
                # Create translation matrix
                shift_x = center_x - cx
                shift_y = center_y - cy
                
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                skeleton = cv2.warpAffine(skeleton, M, (w, h))
        
        # 3. Scale normalization (optional - maintain aspect ratio)
        skeleton = cv2.resize(skeleton, (96, 96))  # Standard size
        
        return skeleton
    
    def _calculate_enhanced_similarity_scores(self):
        """Calculate similarity scores using enhanced multi-level approach"""
        print("Calculating enhanced similarity scores...")
        
        n_skeletons = len(self.processed_skeletons)
        total_comparisons = (n_skeletons * (n_skeletons - 1)) // 2
        
        with tqdm(total=total_comparisons, desc="Computing similarities") as pbar:
            for i in range(n_skeletons):
                for j in range(i + 1, n_skeletons):
                    # Calculate enhanced similarity score
                    score = self._enhanced_skeleton_similarity(
                        self.processed_skeletons[i], 
                        self.processed_skeletons[j],
                        i, j
                    )
                    
                    # Determine if genuine or impostor comparison
                    if self.user_labels[i] == self.user_labels[j]:
                        self.genuine_scores.append(score)
                    else:
                        self.impostor_scores.append(score)
                    
                    pbar.update(1)
        
        print(f"Generated {len(self.genuine_scores)} genuine scores")
        print(f"Generated {len(self.impostor_scores)} impostor scores")
    
    def _enhanced_skeleton_similarity(self, skeleton1, skeleton2, idx1=None, idx2=None):
        """Multi-level similarity combining global and local features"""
        
        # Ensure same dimensions
        if skeleton1.shape != skeleton2.shape:
            min_h, min_w = min(skeleton1.shape[0], skeleton2.shape[0]), min(skeleton1.shape[1], skeleton2.shape[1])
            skel1 = skeleton1[:min_h, :min_w].astype(np.float32)
            skel2 = skeleton2[:min_h, :min_w].astype(np.float32)
        else:
            skel1 = skeleton1.astype(np.float32)
            skel2 = skeleton2.astype(np.float32)
        
        # 1. Local block-based correlation (16x16 blocks)
        h, w = skel1.shape
        block_size = 16
        local_scores = []
        
        for i in range(0, h-block_size, block_size//2):  # Overlapping blocks
            for j in range(0, w-block_size, block_size//2):
                block1 = skel1[i:i+block_size, j:j+block_size].flatten()
                block2 = skel2[i:i+block_size, j:j+block_size].flatten()
                
                if len(block1) > 0 and len(block2) > 0:
                    # Skip empty blocks
                    if np.sum(block1) > 0 or np.sum(block2) > 0:
                        local_corr = np.corrcoef(block1, block2)[0, 1]
                        if not np.isnan(local_corr):
                            local_scores.append(abs(local_corr))
        
        # 2. Ridge pattern similarity (structural)
        structural_score = self._structural_similarity(skel1, skel2)
        
        # 3. Minutiae-based similarity (if enabled)
        minutiae_score = 0
        if self.use_minutiae and idx1 is not None and idx2 is not None:
            minutiae_score = self._minutiae_similarity_cached(idx1, idx2)
        
        # 4. Combine scores with weights
        local_avg = np.mean(local_scores) if local_scores else 0
        
        if self.use_minutiae:
            # Three-way combination
            final_score = 0.4 * local_avg + 0.3 * structural_score + 0.3 * minutiae_score
        else:
            # Two-way combination
            final_score = 0.6 * local_avg + 0.4 * structural_score
        
        return max(0, min(1, final_score))
    
    def _structural_similarity(self, skel1, skel2):
        """Structural similarity focusing on ridge patterns"""
        # Convert to binary for structural analysis
        binary1 = (skel1 > 0.1).astype(np.float32)
        binary2 = (skel2 > 0.1).astype(np.float32)
        
        # Jaccard similarity for binary ridge structure
        intersection = np.sum(binary1 * binary2)
        union = np.sum((binary1 + binary2) > 0)
        jaccard = intersection / union if union > 0 else 0
        
        # Hamming distance component
        hamming = 1 - (np.sum(binary1 != binary2) / binary1.size)
        
        # Normalized cross-correlation
        if np.std(skel1) > 0 and np.std(skel2) > 0:
            ncc = np.corrcoef(skel1.flatten(), skel2.flatten())[0, 1]
            ncc = abs(ncc) if not np.isnan(ncc) else 0
        else:
            ncc = 0
        
        return (jaccard + hamming + ncc) / 3
    
    def _extract_skeleton_minutiae(self, skeleton):
        """Extract ridge endings and bifurcations from skeleton"""
        
        if skeleton is None:
            return []
        
        # Convert skeleton to binary
        binary_skel = (skeleton > 0.1).astype(np.uint8)
        rows, cols = binary_skel.shape
        
        minutiae_points = []
        
        # 8-connected neighborhood for crossing number calculation
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if binary_skel[i, j] == 1:  # Only process ridge pixels
                    
                    # Get 8-connected neighbors
                    neighbors = [
                        binary_skel[i-1, j-1], binary_skel[i-1, j], binary_skel[i-1, j+1],
                        binary_skel[i, j+1], binary_skel[i+1, j+1], binary_skel[i+1, j],
                        binary_skel[i+1, j-1], binary_skel[i, j-1]
                    ]
                    
                    # Calculate crossing number
                    crossing_number = 0
                    for k in range(8):
                        crossing_number += abs(neighbors[k] - neighbors[(k+1) % 8])
                    crossing_number //= 2
                    
                    # Classify minutiae
                    if crossing_number == 1:
                        # Ridge ending
                        minutiae_points.append({
                            'x': j, 'y': i, 'type': 'ending',
                            'angle': self._calculate_minutiae_angle(binary_skel, i, j)
                        })
                    elif crossing_number == 3:
                        # Ridge bifurcation
                        minutiae_points.append({
                            'x': j, 'y': i, 'type': 'bifurcation', 
                            'angle': self._calculate_minutiae_angle(binary_skel, i, j)
                        })
        
        return minutiae_points
    
    def _calculate_minutiae_angle(self, binary_skel, i, j):
        """Calculate the angle of minutiae point"""
        # Simple gradient-based angle calculation
        window_size = 5
        half_window = window_size // 2
        
        y_start = max(0, i - half_window)
        y_end = min(binary_skel.shape[0], i + half_window + 1)
        x_start = max(0, j - half_window)  
        x_end = min(binary_skel.shape[1], j + half_window + 1)
        
        window = binary_skel[y_start:y_end, x_start:x_end]
        
        # Calculate gradients
        gy, gx = np.gradient(window.astype(np.float32))
        angle = np.arctan2(np.mean(gy), np.mean(gx))
        
        return angle
    
    def _minutiae_similarity_cached(self, idx1, idx2):
        """Calculate similarity based on cached minutiae"""
        
        # Get cached minutiae
        key1 = list(self.minutiae_cache.keys())[idx1] if idx1 < len(self.minutiae_cache) else None
        key2 = list(self.minutiae_cache.keys())[idx2] if idx2 < len(self.minutiae_cache) else None
        
        if key1 is None or key2 is None:
            return 0.0
        
        minutiae1 = self.minutiae_cache[key1]
        minutiae2 = self.minutiae_cache[key2]
        
        if len(minutiae1) == 0 or len(minutiae2) == 0:
            return 0.0
        
        # Simple minutiae matching based on spatial proximity
        matched_pairs = 0
        tolerance = 10  # pixels
        angle_tolerance = 0.5  # radians
        
        for m1 in minutiae1:
            for m2 in minutiae2:
                distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                angle_diff = abs(m1['angle'] - m2['angle'])
                
                if (distance < tolerance and 
                    angle_diff < angle_tolerance and 
                    m1['type'] == m2['type']):
                    matched_pairs += 1
                    break  # Each minutiae can match only once
        
        # Calculate similarity score
        max_possible_matches = min(len(minutiae1), len(minutiae2))
        similarity = matched_pairs / max_possible_matches if max_possible_matches > 0 else 0
        
        return similarity
    
    def generate_roc_curve(self):
        """Generate and plot ROC curve"""
        if not self.genuine_scores or not self.impostor_scores:
            print("Error: No scores available for ROC analysis")
            return None
        
        # Create labels (1 for genuine, 0 for impostor)
        genuine_labels = [1] * len(self.genuine_scores)
        impostor_labels = [0] * len(self.impostor_scores)
        
        # Combine scores and labels
        all_scores = self.genuine_scores + self.impostor_scores
        all_labels = genuine_labels + impostor_labels
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Enhanced ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)', fontsize=12)
        plt.ylabel('True Positive Rate (GAR)', fontsize=12)
        plt.title('Enhanced ROC Curve - Fingerprint Biometric System', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\n=== ENHANCED ROC RESULTS ===")
        print(f"AUC Score: {roc_auc:.4f}")
        print(f"Genuine Scores: μ={np.mean(self.genuine_scores):.3f}, σ={np.std(self.genuine_scores):.3f}")
        print(f"Impostor Scores: μ={np.mean(self.impostor_scores):.3f}, σ={np.std(self.impostor_scores):.3f}")
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    def plot_score_distributions(self):
        """Plot genuine and impostor score distributions"""
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(self.impostor_scores, bins=50, alpha=0.6, color='red', 
                label=f'Impostor Scores (n={len(self.impostor_scores)})', density=True)
        plt.hist(self.genuine_scores, bins=50, alpha=0.6, color='green', 
                label=f'Genuine Scores (n={len(self.genuine_scores)})', density=True)
        
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Score Distributions - Genuine vs Impostor (Enhanced)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def calculate_metrics(self, threshold=0.5):
        """Calculate performance metrics at a given threshold"""
        # True positives: genuine scores above threshold
        tp = sum(1 for score in self.genuine_scores if score >= threshold)
        # False negatives: genuine scores below threshold  
        fn = sum(1 for score in self.genuine_scores if score < threshold)
        # True negatives: impostor scores below threshold
        tn = sum(1 for score in self.impostor_scores if score < threshold)
        # False positives: impostor scores above threshold
        fp = sum(1 for score in self.impostor_scores if score >= threshold)
        
        # Calculate rates
        gar = tp / (tp + fn) if (tp + fn) > 0 else 0  # Genuine Accept Rate
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'threshold': threshold,
            'gar': gar,
            'far': far,
            'accuracy': accuracy,
            'tp': tp,
            'fn': fn,
            'tn': tn,
            'fp': fp
        }
        
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Threshold: {threshold:.3f}")
        print(f"Genuine Accept Rate (GAR): {gar:.3f}")
        print(f"False Accept Rate (FAR): {far:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        
        return metrics
    
    def find_eer(self):
        """Find Equal Error Rate (EER) where FAR = FRR"""
        genuine_labels = [1] * len(self.genuine_scores)
        impostor_labels = [0] * len(self.impostor_scores)
        all_scores = self.genuine_scores + self.impostor_scores
        all_labels = genuine_labels + impostor_labels
        
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        frr = 1 - tpr  # False Reject Rate = 1 - True Positive Rate
        
        # Find threshold where FAR ≈ FRR
        eer_index = np.argmin(np.abs(fpr - frr))
        eer = (fpr[eer_index] + frr[eer_index]) / 2
        eer_threshold = thresholds[eer_index]
        
        print(f"\n=== EQUAL ERROR RATE ===")
        print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
        print(f"EER Threshold: {eer_threshold:.3f}")
        
        return {
            'eer': eer,
            'threshold': eer_threshold,
            'far_at_eer': fpr[eer_index],
            'frr_at_eer': frr[eer_index]
        }
    
    def get_summary_report(self):
        """Generate comprehensive performance summary"""
        print("\n" + "="*50)
        print("ENHANCED BIOMETRIC SYSTEM ANALYSIS REPORT")
        print("="*50)
        
        # Basic statistics
        print(f"Dataset Summary:")
        print(f"  Users: {len(set(self.user_labels))}")
        print(f"  Total Templates: {len(self.processed_skeletons)}")
        print(f"  Genuine Comparisons: {len(self.genuine_scores)}")
        print(f"  Impostor Comparisons: {len(self.impostor_scores)}")
        
        # Generate all metrics
        roc_results = self.generate_roc_curve()
        self.plot_score_distributions()
        self.calculate_metrics(threshold=0.6)
        eer_results = self.find_eer()
        
        return {
            'roc_results': roc_results,
            'eer_results': eer_results,
            'dataset_stats': {
                'n_users': len(set(self.user_labels)),
                'n_templates': len(self.processed_skeletons),
                'n_genuine': len(self.genuine_scores),
                'n_impostor': len(self.impostor_scores)
            }
        }
