import cv2
import numpy as np
from config import Config

class ImagePreprocessor:
    """Xử lý tiền xử lý ảnh - NÂNG CẤP với Background Removal"""
    
    @staticmethod
    def resize_image(image, size=Config.MAX_IMAGE_SIZE):
        """Resize ảnh về kích thước chuẩn"""
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def rgb_to_hsv(image):
        """Chuyển đổi RGB sang HSV"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    @staticmethod
    def rgb_to_gray(image):
        """Chuyển đổi RGB sang grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def remove_background_grabcut(image):
        """
        Loại bỏ nền bằng GrabCut (semi-automatic)
        Giả định: vật thể chính ở giữa
        """
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Định nghĩa rectangle chứa vật thể (giữa ảnh)
        h, w = image.shape[:2]
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            return mask2
        except:
            # Fallback: trả về mask toàn 1
            return np.ones(image.shape[:2], dtype=np.uint8)
    
    @staticmethod
    def adaptive_histogram_equalization(image):
        """
        CLAHE - Contrast Limited Adaptive Histogram Equalization
        Tốt hơn histogram equalization thông thường
        """
        if len(image.shape) == 3:
            # Ảnh màu: áp dụng cho kênh V trong HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Ảnh grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    @staticmethod
    def smart_crop(image, mask=None):
        """
        Crop thông minh - Cắt bỏ nền thừa, giữ lại vật thể
        """
        if mask is None:
            return image
        
        # Tìm bounding box của vật thể
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return image
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Thêm padding (10%)
        h, w = image.shape[:2]
        padding_y = int((y_max - y_min) * 0.1)
        padding_x = int((x_max - x_min) * 0.1)
        
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        
        return image[y_min:y_max, x_min:x_max]
    
    @staticmethod
    def normalize_lighting(image):
        """
        Chuẩn hóa ánh sáng - giảm ảnh hưởng của lighting conditions
        """
        if len(image.shape) == 3:
            # Chuyển sang LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge và chuyển lại BGR
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return image
    
    @staticmethod
    def crop_roi(image, x, y, w, h):
        """Cắt vùng quan tâm (Region of Interest)"""
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def convert_to_binary(image, threshold=127):
        """Chuyển ảnh sang đen trắng"""
        gray = ImagePreprocessor.rgb_to_gray(image)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def equalize_histogram(image):
        """Cân bằng histogram để cải thiện contrast"""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    @staticmethod
    def enhance_contrast(image, alpha=1.5, beta=0):
        """Tăng độ tương phản"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def reduce_noise(image, method='gaussian', kernel_size=5):
        """Giảm nhiễu"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        return image
    
    @staticmethod
    def sharpen_image(image):
        """Làm sắc nét ảnh"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def preprocess_pipeline(image_path, enhance=True):
        """
        Pipeline tiền xử lý NÂNG CẤP
        
        Args:
            image_path: đường dẫn ảnh
            enhance: có apply enhancement không (lighting, contrast)
        
        Returns: (original_resized, hsv, gray)
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize về kích thước chuẩn
        image = ImagePreprocessor.resize_image(image)
        
        # Enhancement (optional)
        if enhance:
            # Chuẩn hóa ánh sáng
            image = ImagePreprocessor.normalize_lighting(image)
        
        # Tạo các phiên bản cần thiết
        hsv = ImagePreprocessor.rgb_to_hsv(image)
        gray = ImagePreprocessor.rgb_to_gray(image)
        
        return image, hsv, gray