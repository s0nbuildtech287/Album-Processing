import cv2
import numpy as np
from skimage.feature import hog
from config import Config
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from services.lbp import LBPExtractorOptimized as LBPExtractor

class DeepFeatureExtractor:
    """Extract deep learning features using pre-trained ResNet50"""
    
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        resnet = models.resnet50(pretrained=True)
     
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            features = features.squeeze().cpu().numpy()
            
            features = features / (np.linalg.norm(features) + 1e-7)
            
            return features
            
        except Exception as e:
            print(f"Error extracting deep features from {image_path}: {e}")
            # Return zero vector on error
            return np.zeros(2048)

class FeatureExtractor:
    """Trích xuất đặc trưng ảnh - TỐI ƯU HÓA với Deep Learning"""
    
    def __init__(self, use_deep_features=True):
        self.lbp_extractor = LBPExtractor()
        self.use_deep_features = use_deep_features
        
        # Khởi tạo deep feature extractor nếu được bật
        if self.use_deep_features:
            try:
                print("Initializing Deep Feature Extractor (ResNet50)...")
                self.deep_extractor = DeepFeatureExtractor()
                print("Deep Feature Extractor ready!")
            except Exception as e:
                print(f"Warning: Could not initialize deep features: {e}")
                print("Falling back to traditional features only.")
                self.use_deep_features = False
                self.deep_extractor = None
        else:
            self.deep_extractor = None
    
    def extract_color_histogram(self, hsv_image, bins=(6, 8, 3)):
        h_bins, s_bins, v_bins = bins
          
        hist_h = cv2.calcHist([hsv_image], [0], None, [h_bins], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [s_bins], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [v_bins], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        color_features = np.concatenate([hist_h, hist_s, hist_v])
        
        return color_features
    def extract_color_histogram_manual(hsv_image, bins=(6, 8, 3)):
        h_bins, s_bins, v_bins = bins

        # Tạo histogram rỗng
        hist_h = np.zeros(h_bins, dtype=np.float32)
        hist_s = np.zeros(s_bins, dtype=np.float32)
        hist_v = np.zeros(v_bins, dtype=np.float32)

        height, width, _ = hsv_image.shape
        total_pixels = height * width

        for y in range(height):
            for x in range(width):
                h, s, v = hsv_image[y, x]

                h_idx = int(h * h_bins / 180)
                s_idx = int(s * s_bins / 256)
                v_idx = int(v * v_bins / 256)

                h_idx = min(h_idx, h_bins - 1)
                s_idx = min(s_idx, s_bins - 1)
                v_idx = min(v_idx, v_bins - 1)

                hist_h[h_idx] += 1
                hist_s[s_idx] += 1
                hist_v[v_idx] += 1

        hist_h /= total_pixels
        hist_s /= total_pixels
        hist_v /= total_pixels

        color_features = np.concatenate([hist_h, hist_s, hist_v])

        return color_features

    def extract_lbp_features(self, gray_image):
        """Trích xuất đặc trưng LBP"""
        return self.lbp_extractor.extract_features(gray_image)
    
    def compute_hog_manual(gray_image, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9):
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=1)

        mag = np.sqrt(gx**2 + gy**2)
        angle = np.rad2deg(np.arctan2(gy, gx)) % 180  # hướng 0-180

        h, w = gray_image.shape
        cell_h, cell_w = pixels_per_cell

        n_cells_y = h // cell_h
        n_cells_x = w // cell_w

        orientation_hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_angle = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                hist, _ = np.histogram(
                    cell_angle,
                    bins=orientations,
                    range=(0, 180),
                    weights=cell_mag  # magnitude weight
                )
                orientation_hist[i, j, :] = hist

        block_h, block_w = cells_per_block
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1

        features = []

        for y in range(n_blocks_y):
            for x in range(n_blocks_x):
                block = orientation_hist[y:y+block_h, x:x+block_w, :].ravel()
                # L2 normalize
                block = block / (np.linalg.norm(block) + 1e-7)
                features.extend(block)

        return np.array(features, dtype=np.float32)

    def extract_hog_features(self, gray_image):
        small = cv2.resize(gray_image, (128, 128))
        
        features = hog(
            small,
            orientations=9,
            pixels_per_cell=(16, 16), 
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def extract_all_features(self, image_path, verbose=False):
        from services.preprocessing import ImagePreprocessor
        
        try:
            if verbose:
                print(f"  Extracting traditional features...")
            image, hsv, gray = ImagePreprocessor.preprocess_pipeline(image_path)
            color_features = self.extract_color_histogram(hsv)
            texture_features = self.extract_lbp_features(gray)
            shape_features = self.extract_hog_features(gray)
            
            
            if verbose:
                print(f"       ✓ Color: {len(color_features)} dims")
                print(f"       ✓ Texture: {len(texture_features)} dims")
                print(f"       ✓ Shape: {len(shape_features)} dims")
            
            if self.use_deep_features and self.deep_extractor:
                if verbose:
                    print(f"     Extracting DEEP features (ResNet50)...")
                deep_features = self.deep_extractor.extract_features(image_path)
                if verbose:
                    print(f"       ✓ Deep: {len(deep_features)} dims (semantic features!)")
            else:
                deep_features = np.zeros(2048) 
                if verbose:
                    print(f"      Deep features disabled")
            
            return {
                'color': color_features,
                'texture': texture_features,
                'shape': shape_features,
                'deep': deep_features,  
                'combined': self.combine_features(color_features, texture_features, shape_features, deep_features)
            }
        except Exception as e:
            print(f" Error extracting features from {image_path}: {e}")
            return {
                'color': np.zeros(17),
                'texture': np.zeros(256),
                'shape': np.zeros(324),
                'deep': np.zeros(2048),
                'combined': np.zeros(2645)
            }
    
    def combine_features(self, color_feat, texture_feat, shape_feat, deep_feat):
        """Kết hợp các đặc trưng thành vector tổng hợp bao gồm deep features"""
        color_norm = color_feat / (np.linalg.norm(color_feat) + 1e-7)
        texture_norm = texture_feat / (np.linalg.norm(texture_feat) + 1e-7)
        shape_norm = shape_feat / (np.linalg.norm(shape_feat) + 1e-7)
        deep_norm = deep_feat / (np.linalg.norm(deep_feat) + 1e-7)
        
        combined = np.concatenate([color_norm, texture_norm, shape_norm, deep_norm])
        
        return combined