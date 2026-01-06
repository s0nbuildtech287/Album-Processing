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
        # Load pre-trained ResNet50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and remove the final classification layer
        resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer, keep until avgpool
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Define image preprocessing
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
        """
        Extract 2048-dimensional feature vector from image
        Args:
            image_path: path to image file
        Returns:
            numpy array of shape (2048,)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Flatten and convert to numpy
            features = features.squeeze().cpu().numpy()
            
            # Normalize
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
    
    def extract_all_features(self, image_path, verbose=False):
        """
        Trích xuất tất cả đặc trưng - TỐI ƯU với Deep Learning
        Args:
            image_path: đường dẫn ảnh
            verbose: in log chi tiết
        """
        from services.preprocessing import ImagePreprocessor
        
        try:
            if verbose:
                print(f"    📸 Extracting traditional features...")
            
            # Trích xuất traditional features
            image, hsv, gray = ImagePreprocessor.preprocess_pipeline(image_path)
            color_features = self.extract_color_histogram(hsv)
            texture_features = self.extract_lbp_features(gray)
            shape_features = self.extract_hog_features(gray)
            
            if verbose:
                print(f"       ✓ Color: {len(color_features)} dims")
                print(f"       ✓ Texture: {len(texture_features)} dims")
                print(f"       ✓ Shape: {len(shape_features)} dims")
            
            # Trích xuất deep features nếu có
            if self.use_deep_features and self.deep_extractor:
                if verbose:
                    print(f"    🧠 Extracting DEEP features (ResNet50)...")
                deep_features = self.deep_extractor.extract_features(image_path)
                if verbose:
                    print(f"       ✓ Deep: {len(deep_features)} dims (semantic features!)")
            else:
                deep_features = np.zeros(2048)  # Placeholder
                if verbose:
                    print(f"    ⚠️  Deep features disabled")
            
            return {
                'color': color_features,
                'texture': texture_features,
                'shape': shape_features,
                'deep': deep_features,  # NEW: Deep learning features
                'combined': self.combine_features(color_features, texture_features, shape_features, deep_features)
            }
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            # Return dummy features if error
            return {
                'color': np.zeros(17),
                'texture': np.zeros(256),
                'shape': np.zeros(324),
                'deep': np.zeros(2048),
                'combined': np.zeros(2645)  # Updated size
            }
    
    def combine_features(self, color_feat, texture_feat, shape_feat, deep_feat):
        """Kết hợp các đặc trưng thành vector tổng hợp bao gồm deep features"""
        # Chuẩn hóa mỗi loại đặc trưng
        color_norm = color_feat / (np.linalg.norm(color_feat) + 1e-7)
        texture_norm = texture_feat / (np.linalg.norm(texture_feat) + 1e-7)
        shape_norm = shape_feat / (np.linalg.norm(shape_feat) + 1e-7)
        deep_norm = deep_feat / (np.linalg.norm(deep_feat) + 1e-7)
        
        # Ghép lại
        combined = np.concatenate([color_norm, texture_norm, shape_norm, deep_norm])
        
        return combined