import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

class MinutiaeROCAnalyzer:
    def __init__(self, users_minutiae, distance_threshold=20, angle_threshold=30, match_threshold=15):
        """
        Args:
            users_minutiae: dict {user_id: [ [minutiae_of_sample1], [minutiae_of_sample2], ... ]}
                            Each minutiae list is a list of minutiae [(x,y), type, angle].
            distance_threshold: max pixel distance to consider minutiae a match
            angle_threshold: max angle difference in degrees to consider a match
            match_threshold: minimum matched minutiae count to classify as genuine
        """
        self.users_minutiae = users_minutiae
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.match_threshold = match_threshold
        self.genuine_scores = []
        self.impostor_scores = []
        self.all_labels = []
        self.all_scores = []
        self.roc_data = {}
        
    

    