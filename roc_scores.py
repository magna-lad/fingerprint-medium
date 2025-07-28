import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from itertools import combinations
import pandas as pd
from tqdm import tqdm

class MinutiaeROCAnalyzer:
    def __init__(self, users_data, distance_threshold=20, angle_threshold=30):
        """
        Initialize ROC analyzer for minutiae-based fingerprint recognition
        
        Args:
            users_data: Dictionary with user minutiae data 
            {
            user1:{
                'finger':finger1,finger2,.....,
                'minutiae':minutiae1,minutiae2,...
                },
            user2:{
                'finger':finger1,finger2,.....,
                'minutiae':minutiae1,minutiae2,...
                }
            
                
            distance_threshold: Maximum distance for minutiae matching (pixels)
            angle_threshold: Maximum angle difference for minutiae matching (degrees)
        """
        self.users_data = users_data
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.genuine_scores = []
        self.impostor_scores = []
        self.roc_data = {}
        
    def calculate_minutiae_similarity(self, minutiae1, minutiae2):
        """
        Calculate similarity between two sets of minutiae points
        
        Args:
            minutiae1, minutiae2: Lists of tuples [(x, y), type, angle}
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if not minutiae1 or not minutiae2:
            return 0.0
        
        # Convert to numpy arrays for easier processing
        m1 = np.array(minutiae1)
        m2 = np.array(minutiae2)
        
        # Extract coordinates, angles, and types
        coords1 = (m1[0][0].astype(float))
        coords2 = (m2[0][0].astype(float))
        angles1 = (m1[2].astype(float))
        angles2 = (m2[2].astype(float))
        types1 = (m1[1].astype(int))
        types2 = (m2[1].astype(int))
        
        matches = 0
        total_possible = min(len(minutiae1), len(minutiae2))
        
        if total_possible == 0:
            return 0.0
        
        # For each minutiae in set 1, find best match in set 2
        used_indices = set()
        
        for i in range(len(coords1)):
            best_match_idx = -1
            best_score = float('inf')
            
            for j in range(len(coords2)):
                if j in used_indices:
                    continue
                
                # Check if types match
                if types1[i] != types2[j]:
                    continue
                
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum((coords1[i] - coords2[j]) ** 2))
                
                # Calculate angle difference (handle circular nature)
                angle_diff = abs(angles1[i] - angles2[j])
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                # Check if within thresholds
                if distance <= self.distance_threshold and angle_diff <= self.angle_threshold:
                    if distance < best_score:
                        best_score = distance
                        best_match_idx = j
            
            # If we found a valid match
            if best_match_idx != -1:
                matches += 1
                used_indices.add(best_match_idx)
        
        # Calculate similarity score
        similarity = matches / max(len(minutiae1), len(minutiae2))
        return similarity
    




    def calculate_advanced_similarity(self, minutiae1, minutiae2):
        """
        Advanced similarity calculation using spatial relationships
        """
        if not minutiae1 or not minutiae2:
            return 0.0
        
        # Basic minutiae matching
        basic_sim = self.calculate_minutiae_similarity(minutiae1, minutiae2)
        
        # Add spatial relationship analysis
        spatial_sim = self.analyze_spatial_relationships(minutiae1, minutiae2)
        
        # Combine scores (weighted average)
        final_score = 0.7 * basic_sim + 0.3 * spatial_sim
        return final_score
    




    def analyze_spatial_relationships(self, minutiae1, minutiae2):
        """
        Analyze spatial relationships between minutiae points
        """
        if len(minutiae1) < 2 or len(minutiae2) < 2:
            return 0.0
        
        # Calculate distances between all pairs of minutiae in each set
        coords1 = np.array([(m[0], m[1]) for m in minutiae1])
        coords2 = np.array([(m[0], m[1]) for m in minutiae2])
        
        # Distance matrices
        dist_matrix1 = cdist(coords1, coords1)
        dist_matrix2 = cdist(coords2, coords2)
        
        # Compare distance patterns (simplified approach)
        # In practice, you'd use more sophisticated geometric hashing
        similar_patterns = 0
        total_patterns = 0
        
        for i in range(min(5, len(minutiae1))):  # Limit to avoid computational explosion
            for j in range(i + 1, min(5, len(minutiae1))):
                dist1 = dist_matrix1[i, j]
                
                # Find similar distance in second set
                for x in range(len(minutiae2)):
                    for y in range(x + 1, len(minutiae2)):
                        dist2 = dist_matrix2[x, y]
                        if abs(dist1 - dist2) < 10:  # Allow some tolerance
                            similar_patterns += 1
                            break
                    else:
                        continue
                    break
                
                total_patterns += 1
        
        if total_patterns == 0:
            return 0.0
        
        return similar_patterns / total_patterns
    



    def compute_all_scores(self, use_advanced=True):
        """
        Compute all genuine and impostor scores
        """
        print("Computing biometric scores...")
        self.genuine_scores = []
        self.impostor_scores = []
        
        user_ids = list(self.users_data.keys())
        
        # Calculate genuine scores (same user, different samples)
        print("Computing genuine scores...")
        for user_id in tqdm(user_ids, desc="Genuine scores"):
            user_minutiae = self.users_data[user_id].get('minutiae', [])
            
            # Compare all pairs of samples from the same user
            for i in range(len(user_minutiae)):
                for j in range(i + 1, len(user_minutiae)):
                    if use_advanced:
                        score = self.calculate_advanced_similarity(
                            user_minutiae[i], user_minutiae[j]
                        )
                    else:
                        score = self.calculate_minutiae_similarity(
                            user_minutiae[i], user_minutiae[j]
                        )
                    self.genuine_scores.append(score)
        
        # Calculate impostor scores (different users)
        print("Computing impostor scores...")
        for i, user1 in enumerate(tqdm(user_ids, desc="Impostor scores")):
            for j, user2 in enumerate(user_ids):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                user1_minutiae = self.users_data[user1].get('minutiae', [])
                user2_minutiae = self.users_data[user2].get('minutiae', [])
                
                # Compare first sample of each user (to limit computation)
                if user1_minutiae and user2_minutiae:
                    if use_advanced:
                        score = self.calculate_advanced_similarity(
                            user1_minutiae[0], user2_minutiae[0]
                        )
                    else:
                        score = self.calculate_minutiae_similarity(
                            user1_minutiae[0], user2_minutiae[0]
                        )
                    self.impostor_scores.append(score)
        
        print(f"Computed {len(self.genuine_scores)} genuine scores")
        print(f"Computed {len(self.impostor_scores)} impostor scores")
    





    def generate_roc_curve(self):
        """
        Generate ROC curve data
        """
        if not self.genuine_scores or not self.impostor_scores:
            raise ValueError("Must compute scores first using compute_all_scores()")
        
        # Prepare data for ROC curve
        # Genuine scores should have label 1, impostor scores label 0
        y_true = [1] * len(self.genuine_scores) + [0] * len(self.impostor_scores)
        y_scores = self.genuine_scores + self.impostor_scores
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        
        # Store ROC data
        self.roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
        
        return self.roc_data
    


    def calculate_eer(self):
        """
        Calculate Equal Error Rate (EER)
        """
        if not self.roc_data:
            self.generate_roc_curve()
        
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        
        # Find point where FPR ≈ FNR (1 - TPR)
        fnr = 1 - tpr
        eer_index = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        
        return eer
    































    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve
        """
        if not self.roc_data:
            self.generate_roc_curve()
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(self.roc_data['fpr'], self.roc_data['tpr'], 
                color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {self.roc_data["auc"]:.3f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        # Calculate and plot EER point
        eer = self.calculate_eer()
        plt.plot([eer], [1-eer], 'ro', markersize=8, 
                label=f'EER = {eer:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1 - FRR)')
        plt.title('ROC Curve - Minutiae-based Fingerprint Recognition')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_score_distributions(self, save_path=None):
        """
        Plot genuine and impostor score distributions
        """
        if not self.genuine_scores or not self.impostor_scores:
            raise ValueError("Must compute scores first")
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(self.impostor_scores, bins=50, alpha=0.7, 
                label='Impostor Scores', color='red', density=True)
        plt.hist(self.genuine_scores, bins=50, alpha=0.7, 
                label='Genuine Scores', color='green', density=True)
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Score Distributions - Genuine vs Impostor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.axvline(np.mean(self.genuine_scores), color='green', 
                   linestyle='--', label=f'Genuine Mean: {np.mean(self.genuine_scores):.3f}')
        plt.axvline(np.mean(self.impostor_scores), color='red', 
                   linestyle='--', label=f'Impostor Mean: {np.mean(self.impostor_scores):.3f}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_performance_metrics(self):
        """
        Get comprehensive performance metrics
        """
        if not self.roc_data:
            self.generate_roc_curve()
        
        eer = self.calculate_eer()
        
        metrics = {
            'auc': self.roc_data['auc'],
            'eer': eer,
            'genuine_mean': np.mean(self.genuine_scores),
            'genuine_std': np.std(self.genuine_scores),
            'impostor_mean': np.mean(self.impostor_scores),
            'impostor_std': np.std(self.impostor_scores),
            'total_genuine': len(self.genuine_scores),
            'total_impostor': len(self.impostor_scores)
        }
        
        return metrics
    
    def print_performance_report(self):
        """
        Print detailed performance report
        """
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*50)
        print("MINUTIAE-BASED FINGERPRINT RECOGNITION REPORT")
        print("="*50)
        
        print(f"Dataset Statistics:")
        print(f"  - Users: {len(self.users_data)}")
        print(f"  - Genuine comparisons: {metrics['total_genuine']}")
        print(f"  - Impostor comparisons: {metrics['total_impostor']}")
        
        print(f"\nPerformance Metrics:")
        print(f"  - AUC (Area Under Curve): {metrics['auc']:.4f}")
        print(f"  - EER (Equal Error Rate): {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
        
        print(f"\nScore Statistics:")
        print(f"  - Genuine scores: {metrics['genuine_mean']:.4f} ± {metrics['genuine_std']:.4f}")
        print(f"  - Impostor scores: {metrics['impostor_mean']:.4f} ± {metrics['impostor_std']:.4f}")
        
        print(f"\nConfiguration:")
        print(f"  - Distance threshold: {self.distance_threshold} pixels")
        print(f"  - Angle threshold: {self.angle_threshold} degrees")
        
        print("="*50)

# Usage integration with your existing code
def perform_roc_analysis(users_data):
    """
    Perform complete ROC analysis on minutiae data
    """
    # Initialize ROC analyzer
    roc_analyzer = MinutiaeROCAnalyzer(
        users_data, 
        distance_threshold=20, 
        angle_threshold=30
    )
    
    # Compute all similarity scores
    roc_analyzer.compute_all_scores(use_advanced=True)
    
    # Generate ROC curve
    roc_data = roc_analyzer.generate_roc_curve()
    
    # Print performance report
    roc_analyzer.print_performance_report()
    
    # Plot results
    roc_analyzer.plot_roc_curve()
    roc_analyzer.plot_score_distributions()
    
    return roc_analyzer

