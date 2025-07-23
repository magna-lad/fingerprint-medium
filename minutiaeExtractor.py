import numpy as np
import cv2

class minutiaeExtractor:
    def __init__(self, user_image):
        '''
        Extract minutiae from skeletonized fingerprint image
        '''
        self.xcord = []
        self.ycord = []
        self.angle = []
        self.type = []  # 0 for ridge ending, 1 for bifurcation
        self.user_image = user_image
        self.block = 5  # will find minutiae within 5 pixels

    def get_crossing_number(self, skeleton, i, j):
        """Calculate crossing number for minutiae detection"""
        # 8-connected neighborhood in clockwise order
        neighbors = [
            skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
            skeleton[i, j+1], skeleton[i+1, j+1], skeleton[i+1, j],
            skeleton[i+1, j-1], skeleton[i, j-1]
        ]
        
        # Convert to binary (0 or 1)
        neighbors = [1 if pixel > 0 else 0 for pixel in neighbors]
        
        # Calculate crossing number using the formula
        crossing_number = 0
        for k in range(8):
            crossing_number += abs(neighbors[k] - neighbors[(k+1) % 8])
        
        return crossing_number // 2

    def calculate_angle(self, skeleton, i, j):
        """Calculate the orientation angle of the minutiae"""
        # Define a local window around the minutiae point
        window_size = 7
        half_size = window_size // 2
        
        # Ensure we don't go out of bounds
        y_start = max(0, i - half_size)
        y_end = min(skeleton.shape[0], i + half_size + 1)
        x_start = max(0, j - half_size)
        x_end = min(skeleton.shape[1], j + half_size + 1)
        
        local_patch = skeleton[y_start:y_end, x_start:x_end]
        
        # Find all ridge pixels in the local patch
        ridge_y, ridge_x = np.where(local_patch > 0)
        
        if len(ridge_y) < 3:
            return 0
        
        # Convert to actual coordinates
        ridge_points = np.column_stack((ridge_x, ridge_y))
        
        # Calculate the principal direction using covariance
        if len(ridge_points) > 1:
            centroid = np.mean(ridge_points, axis=0)
            centered_points = ridge_points - centroid
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Get eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Principal direction is the eigenvector with largest eigenvalue
            principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Calculate angle in degrees
            angle = np.arctan2(principal_direction[1], principal_direction[0])
            return np.degrees(angle) % 180
        
        return 0

    def remove_border_minutiae(self, width, height, border_distance=10):
        """Remove minutiae that are too close to image borders"""
        valid_indices = []
        # Make border distance proportional to image size
        border_distance = max(5, min(width, height) // 20)

        
        for i in range(len(self.xcord)):
            x, y = self.xcord[i], self.ycord[i]
            if (border_distance <= x < width - border_distance and 
                border_distance <= y < height - border_distance):
                valid_indices.append(i)
        
        # Keep only valid minutiae
        self.xcord = [self.xcord[i] for i in valid_indices]
        self.ycord = [self.ycord[i] for i in valid_indices]
        self.angle = [self.angle[i] for i in valid_indices]
        self.type = [self.type[i] for i in valid_indices]

    def extract(self):
        """Extract minutiae from the skeletonized fingerprint image"""
        h, w = self.user_image.shape
        
        # Clear previous results
        self.xcord.clear()
        self.ycord.clear()
        self.angle.clear()
        self.type.clear()
        
        # Scan through the skeleton image
        for i in range(1, h - 1):  # Skip border pixels
            for j in range(1, w - 1):
                if self.user_image[i, j] > 0:  # If pixel is part of ridge
                    cn = self.get_crossing_number(self.user_image, i, j)
                    
                    # Ridge ending: crossing number = 1
                    if cn == 1:
                        angle = self.calculate_angle(self.user_image, i, j)
                        self.xcord.append(j)
                        self.ycord.append(i)
                        self.angle.append(angle)
                        self.type.append(0)  # 0 for ridge ending
                    
                    # Bifurcation: crossing number = 3
                    elif cn == 3:
                        angle = self.calculate_angle(self.user_image, i, j)
                        self.xcord.append(j)
                        self.ycord.append(i)
                        self.angle.append(angle)
                        self.type.append(1)  # 1 for bifurcation
        
        # Remove minutiae too close to borders
        self.remove_border_minutiae(w, h)
        
        # Return list of tuples (x, y, angle, type) as requested
        minutiae_points = [(self.xcord[i], self.ycord[i], self.angle[i], self.type[i]) 
                           for i in range(len(self.xcord))]
        return minutiae_points
