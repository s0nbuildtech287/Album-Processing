import numpy as np
import cv2
from config import Config

class LBPExtractorOptimized:
    """Trích xuất đặc trưng Local Binary Pattern - TỐI ƯU HÓA"""
    
    def __init__(self, radius=Config.LBP_RADIUS, points=Config.LBP_POINTS):
        self.radius = radius
        self.points = points
        self.n_bins = min(256, 2 ** points)
    
    def compute_lbp_simple(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                
                lbp[i-1, j-1] = code
        
        return lbp
    
    def compute_lbp_histogram(self, image, normalize=True):
        """Tính histogram LBP nhanh"""
        lbp_image = self.compute_lbp_simple(image)
        
        hist, _ = np.histogram(lbp_image.ravel(), 
                               bins=256,  
                               range=(0, 256))
        
        if normalize:
            hist = hist.astype(float)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
        
        return hist
    
    def extract_features(self, image):
        """Trích xuất vector đặc trưng LBP"""
        return self.compute_lbp_histogram(image)