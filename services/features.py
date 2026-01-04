import cv2
import numpy as np
from skimage.feature import hog
from config import Config


from services.lbp import LBPExtractorOptimized as LBPExtractor

class FeatureExtractor:
    """Trích xuất đặc trưng ảnh - TỐI ƯU HÓA"""
    
    def __init__(self):
        self.lbp_extractor = LBPExtractor()
    
    def extract_color_histogram(self, hsv_image, bins=(6, 8, 3)):
        """
        Trích xuất histogram màu HSV - giảm bins để nhanh hơn
        """
        h_bins, s_bins, v_bins = bins
        
        # Tính histogram cho mỗi kênh
        hist_h = cv2.calcHist([hsv_image], [0], None, [h_bins], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [s_bins], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [v_bins], [0, 256])
        
        # Chuẩn hóa
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Ghép thành vector
        color_features = np.concatenate([hist_h, hist_s, hist_v])
        
        return color_features
    
    def extract_lbp_features(self, gray_image):
        """Trích xuất đặc trưng LBP"""
        return self.lbp_extractor.extract_features(gray_image)
    
    def extract_hog_features(self, gray_image):
        """
        Trích xuất đặc trưng HOG - tối ưu
        """
        # Resize nhỏ hơn để nhanh hơn
        small = cv2.resize(gray_image, (128, 128))
        
        features = hog(
            small,
            orientations=9,
            pixels_per_cell=(16, 16),  # Tăng cell size để nhanh hơn
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        
        # Chuẩn hóa
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def extract_all_features(self, image_path):
        """
        Trích xuất tất cả đặc trưng - TỐI ƯU
        """
        from services.preprocessing import ImagePreprocessor
        
        try:
            # Tiền xử lý
            image, hsv, gray = ImagePreprocessor.preprocess_pipeline(image_path)
            
            # Trích xuất đặc trưng
            color_features = self.extract_color_histogram(hsv)
            texture_features = self.extract_lbp_features(gray)
            shape_features = self.extract_hog_features(gray)
            
            return {
                'color': color_features,
                'texture': texture_features,
                'shape': shape_features,
                'combined': self.combine_features(color_features, texture_features, shape_features)
            }
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            # Return dummy features if error
            return {
                'color': np.zeros(17),
                'texture': np.zeros(256),
                'shape': np.zeros(324),
                'combined': np.zeros(597)
            }
    
    def combine_features(self, color_feat, texture_feat, shape_feat):
        """Kết hợp các đặc trưng thành vector tổng hợp"""
        # Chuẩn hóa mỗi loại đặc trưng
        color_norm = color_feat / (np.linalg.norm(color_feat) + 1e-7)
        texture_norm = texture_feat / (np.linalg.norm(texture_feat) + 1e-7)
        shape_norm = shape_feat / (np.linalg.norm(shape_feat) + 1e-7)
        
        # Ghép lại
        combined = np.concatenate([color_norm, texture_norm, shape_norm])
        
        return combined