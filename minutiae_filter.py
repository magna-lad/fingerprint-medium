import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# update the code for the new datastructure
class minutiae_filter:
    def __init__(self, skeleton, minutiae, mask):
        self.skeleton = skeleton        # list of 2D arrays
        self.minutiae = minutiae       # list of [(x, y), type, angle]
        self.mask = mask               # list of binary masks (0 and 255 or 0 and 1)
 
    def filter_all(self, min_distance=15, boundary_thickness=15, angle_threshold=30):
        filtered_fingers = []
        filtered_minutiae_sets = []

        for skeleton, minutiae, mask in (self.skeleton, self.minutiae, self.mask):
            h, w = skeleton.shape
            filtered = []

            

            # Step 1: Binarize mask
            mask_bin = (mask > 0).astype(np.uint8)

            # Step 2: Get mask boundary using morphological gradient
            kernel = np.ones((3, 3), np.uint8)
            mask_boundary = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, kernel)

            # Step 3: Thicken mask boundary
            if boundary_thickness > 1:
                mask_boundary = cv2.dilate(mask_boundary, kernel, iterations=boundary_thickness)

            # Step 4: Create image border mask
            img_border = np.zeros_like(mask_boundary)
            img_border[:(boundary_thickness+10), :] = 1
            img_border[-(boundary_thickness+10):, :] = 1
            img_border[:, :(boundary_thickness+10)] = 1
            img_border[:, -(boundary_thickness+10):] = 1

            # Step 5: Combine both boundaries
            combined_boundary = np.clip(mask_boundary + img_border, 0, 1).astype(np.uint8)

            for pt in minutiae:  # [(x, y), type, angle]
                x, y = pt[0]

                # setting type: 'bifurcation'-1
                        #   'ending'-0  
                if pt[1] == 'bifurcation':
                    pt[1] =1
                elif pt[1] == 'ending':
                    pt[1] = 0

                # Skip if out of bounds
                if not (0 <= x < w and 0 <= y < h):
                    continue

                # 1. Skip if on boundary
                if combined_boundary[y, x] == 1:
                    continue

                # 2. Ridge check
                if not self._is_on_valid_ridge(skeleton, x, y):
                    skeleton[y, x] = 0
                    continue

                # 3. Proximity check
                too_close = False
                for other in filtered:
                    ox, oy = other[0]
                    dist = ((x - ox)**2 + (y - oy)**2)**0.5
                    if dist < min_distance:
                        too_close = True
                        break
                if too_close:
                    skeleton[y, x] = 0
                    continue

                # 4. Ridge Orientation Consistency Check
                angle_ok = self._is_orientation_consistent(pt, minutiae, angle_threshold)
                if not angle_ok:
                    continue
                
               
                filtered.append(pt)

            filtered_fingers.append(skeleton)
            filtered_minutiae_sets.append(filtered)

        return filtered_fingers, filtered_minutiae_sets

    def _is_on_valid_ridge(self, img, x, y):
        y_min = max(0, y - 1)
        y_max = min(img.shape[0], y + 2)
        x_min = max(0, x - 1)
        x_max = min(img.shape[1], x + 2)

        neighbors = img[y_min:y_max, x_min:x_max]
        return np.sum(neighbors) > 2

    def _is_orientation_consistent(self, pt, all_minutiae, threshold_deg=30, radius=20):
        """
        Compares current angle with surrounding angles within radius.
        Removes minutiae if orientation is inconsistent.
        """
        x, y = pt[0]
        angle = pt[2]
        nearby_angles = []

        for other in all_minutiae:
            ox, oy = other[0]
            if (ox == x and oy == y):
                continue
            dist = np.hypot(ox - x, oy - y)
            if dist <= radius:
                nearby_angles.append(other[2])

        if not nearby_angles:
            return True  # No neighbors, allow

        # Convert to unit circle
        a1 = np.deg2rad(angle)
        sum_x = np.cos(a1)
        sum_y = np.sin(a1)

        for a in nearby_angles:
            rad = np.deg2rad(a)
            sum_x += np.cos(rad)
            sum_y += np.sin(rad)

        mean_angle = np.arctan2(sum_y, sum_x)
        mean_angle_deg = np.rad2deg(mean_angle) % 360

        # Compute circular distance
        diff = np.abs((angle - mean_angle_deg + 180) % 360 - 180)
        return diff <= threshold_deg
