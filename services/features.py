import cv2
import numpy as np
from skimage.feature import hog
from config import Config
from services.lbp import LBPExtractorOptimized as LBPExtractor

class FeatureExtractor:
    """Trích xuất đặc trưng NÂNG CẤP - Object-aware + Multi-scale"""
    
    def __init__(self):
        self.lbp_extractor = LBPExtractor()
    
    def detect_foreground_mask(self, image):
        """
        PHÁT HIỆN VẬT THỂ CHÍNH - Loại bỏ nền
        Sử dụng: GrabCut + Edge detection + Saliency
        """
        h, w = image.shape[:2]
        
        # Method 1: Saliency Map (phát hiện vùng quan trọng)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliency_map) = saliency.computeSaliency(image)
        saliency_map = (saliency_map * 255).astype("uint8")
        
        # Threshold để tạo mask
        _, mask = cv2.threshold(saliency_map, 100, 255, cv2.THRESH_BINARY)
        
        # Morphology để làm mịn mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fallback: nếu mask quá nhỏ/quá lớn, dùng center ellipse
        foreground_ratio = np.sum(mask > 0) / (h * w)
        if foreground_ratio < 0.1 or foreground_ratio > 0.95:
            # Dùng ellipse mask trung tâm (80%)
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes = (int(w * 0.40), int(h * 0.40))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def extract_color_histogram(self, hsv_image, mask=None, bins=(8, 12, 3)):
        """
        Histogram màu với FOREGROUND MASK
        Chỉ tính màu của vật thể, bỏ qua nền
        """
        h_bins, s_bins, v_bins = bins
        
        # Tính histogram CHỈ trong vùng mask
        hist_h = cv2.calcHist([hsv_image], [0], mask, [h_bins], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], mask, [s_bins], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], mask, [v_bins], [0, 256])
        
        # Normalize
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Thêm: Dominant color (màu chủ đạo)
        # Lấy top-3 màu xuất hiện nhiều nhất
        masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        pixels = masked_hsv[mask > 0].reshape(-1, 3)
        
        if len(pixels) > 0:
            # Quantize colors (giảm số màu xuống 32 để dễ tìm dominant)
            pixels_quant = (pixels // 8) * 8
            unique, counts = np.unique(pixels_quant, axis=0, return_counts=True)
            
            # Top-3 màu
            top_indices = np.argsort(counts)[-3:][::-1]
            dominant_colors = unique[top_indices].flatten() / 255.0  # Normalize 0-1
        else:
            dominant_colors = np.zeros(9)  # 3 colors x 3 channels
        
        return np.concatenate([hist_h, hist_s, hist_v, dominant_colors])
    
    def extract_edge_features(self, gray_image, mask=None):
        """
        EDGE FEATURES - Phát hiện viền vật thể
        Quan trọng cho shape recognition
        """
        # Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Chỉ lấy edge trong vùng mask
        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Edge histogram (8 directions)
        # Tính hướng của edge
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính góc (0-360 độ)
        angles = np.arctan2(sobely, sobelx) * 180 / np.pi
        angles[angles < 0] += 360
        
        # Apply mask
        if mask is not None:
            angles = angles * (mask > 0)
        
        # Histogram 8 bins (mỗi bin = 45 độ)
        edge_hist, _ = np.histogram(angles[edges > 0], bins=8, range=(0, 360))
        edge_hist = edge_hist / (np.sum(edge_hist) + 1e-10)
        
        # Edge density
        edge_density = np.sum(edges > 0) / (edges.size + 1e-10)
        
        return np.concatenate([edge_hist, [edge_density]])
    
    def extract_lbp_features(self, gray_image, mask=None):
        """LBP với mask"""
        features = self.lbp_extractor.extract_features(gray_image)
        return features
    
    def extract_hog_features(self, gray_image, mask=None):
        """
        HOG NÂNG CẤP - Multi-scale + Masked
        """
        small = cv2.resize(gray_image, (128, 128))
        
        # Resize mask tương ứng
        if mask is not None:
            mask_small = cv2.resize(mask, (128, 128))
            # Apply mask: chuyển vùng ngoài mask = 0
            small = cv2.bitwise_and(small, small, mask=mask_small)
        
        # HOG features
        features = hog(
            small,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        
        # Multi-scale: thêm HOG scale nhỏ hơn (chi tiết hơn)
        tiny = cv2.resize(gray_image, (64, 64))
        if mask is not None:
            mask_tiny = cv2.resize(mask, (64, 64))
            tiny = cv2.bitwise_and(tiny, tiny, mask=mask_tiny)
        
        features_tiny = hog(
            tiny,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        
        # Combine multi-scale
        combined = np.concatenate([features, features_tiny])
        
        return combined / (np.linalg.norm(combined) + 1e-7)
    
    def extract_spatial_layout(self, hsv_image, mask, grid_size=4):
        """
        SPATIAL LAYOUT - Phân bố màu sắc theo vị trí
        Ví dụ: Trời xanh ở trên, cỏ xanh ở dưới
        """
        h, w = hsv_image.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        layout_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Crop cell
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                cell_hsv = hsv_image[y1:y2, x1:x2]
                cell_mask = mask[y1:y2, x1:x2] if mask is not None else None
                
                # Color histogram cho cell này
                hist_h = cv2.calcHist([cell_hsv], [0], cell_mask, [4], [0, 180])
                hist_s = cv2.calcHist([cell_hsv], [1], cell_mask, [4], [0, 256])
                
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                hist_s = cv2.normalize(hist_s, hist_s).flatten()
                
                layout_features.extend(hist_h)
                layout_features.extend(hist_s)
        
        return np.array(layout_features)
    
    def extract_all_features(self, image_path):
        """
        PIPELINE NÂNG CẤP - Extract ALL features với foreground detection
        """
        from services.preprocessing import ImagePreprocessor
        
        try:
            # 1. Preprocessing
            image, hsv, gray = ImagePreprocessor.preprocess_pipeline(image_path)
            
            # 2. PHÁT HIỆN FOREGROUND (vật thể chính)
            mask = self.detect_foreground_mask(image)
            
            # 3. EXTRACT FEATURES với mask
            # Color: histogram + dominant colors
            color_features = self.extract_color_histogram(hsv, mask, Config.HSV_BINS)
            
            # Texture: LBP
            texture_features = self.extract_lbp_features(gray, mask)
            
            # Shape: HOG multi-scale + Edge
            shape_hog = self.extract_hog_features(gray, mask)
            edge_features = self.extract_edge_features(gray, mask)
            shape_features = np.concatenate([shape_hog, edge_features])
            
            # Spatial layout (phân bố không gian)
            spatial_features = self.extract_spatial_layout(hsv, mask, grid_size=4)
            
            # 4. COMBINE
            return {
                'color': color_features,
                'texture': texture_features,
                'shape': shape_features,
                'spatial': spatial_features,
                'combined': self.combine_features(
                    color_features, 
                    texture_features, 
                    shape_features,
                    spatial_features
                )
            }
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def combine_features(self, color_feat, texture_feat, shape_feat, spatial_feat):
        """Combine + Normalize tất cả features"""
        color_norm = color_feat / (np.linalg.norm(color_feat) + 1e-7)
        texture_norm = texture_feat / (np.linalg.norm(texture_feat) + 1e-7)
        shape_norm = shape_feat / (np.linalg.norm(shape_feat) + 1e-7)
        spatial_norm = spatial_feat / (np.linalg.norm(spatial_feat) + 1e-7)
        
        return np.concatenate([
            color_norm * 0.6,    # Weight color
            texture_norm * 0.8,  # Weight texture
            shape_norm * 1.0,    # Weight shape (cao nhất)
            spatial_norm * 0.4   # Weight spatial
        ])