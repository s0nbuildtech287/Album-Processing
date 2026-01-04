import cv2
import numpy as np
from config import Config

class ImagePreprocessor:
    """Xử lý tiền xử lý ảnh"""
    
    @staticmethod
    def resize_image(image, size=Config.MAX_IMAGE_SIZE):
        """Resize ảnh về kích thước chuẩn"""
        # Dùng INTER_LINEAR nhanh hơn INTER_AREA
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
            # Ảnh màu: áp dụng cho kênh V trong HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Ảnh grayscale
            return cv2.equalizeHist(image)
    
    @staticmethod
    def enhance_contrast(image, alpha=1.5, beta=0):
        """
        Tăng độ tương phản
        alpha: contrast control (1.0-3.0)
        beta: brightness control (0-100)
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def reduce_noise(image, method='gaussian', kernel_size=5):
        """
        Giảm nhiễu
        method: 'gaussian', 'median', 'bilateral'
        """
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
    def preprocess_pipeline(image_path):
        """
        Pipeline tiền xử lý chuẩn cho hệ thống
        Returns: (original_resized, hsv, gray)
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize về kích thước chuẩn
        image = ImagePreprocessor.resize_image(image)
        
        # Tạo các phiên bản cần thiết
        hsv = ImagePreprocessor.rgb_to_hsv(image)
        gray = ImagePreprocessor.rgb_to_gray(image)
        
        return image, hsv, gray