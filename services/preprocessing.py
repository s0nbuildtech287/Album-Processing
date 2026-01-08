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
    
    # @staticmethod
    # def rgb_to_hsv(image):
    #     """Chuyển đổi RGB sang HSV"""
    #     return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # @staticmethod
    # def rgb_to_gray(image):
    #     """Chuyển đổi RGB sang grayscale"""
    #     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def rgb_to_gray(image):
        B = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        R = image[:, :, 2].astype(np.float32)
        gray = 0.114 * B + 0.587 * G + 0.299 * R
        return gray.astype(np.uint8)

    @staticmethod
    def rgb_to_hsv(image):
        img = image.astype(np.float32) / 255.0
        B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        maxc = np.maximum(np.maximum(R, G), B)
        minc = np.minimum(np.minimum(R, G), B)
        diff = maxc - minc

        V = maxc
        S = np.zeros_like(maxc)
        mask = maxc != 0
        S[mask] = diff[mask] / maxc[mask]

        H = np.zeros_like(maxc)
        mask_r = (maxc == R) & (diff != 0)
        mask_g = (maxc == G) & (diff != 0)
        mask_b = (maxc == B) & (diff != 0)

        H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / diff[mask_r]) + 360) % 360
        H[mask_g] = (60 * ((B[mask_g] - R[mask_g]) / diff[mask_g]) + 120)
        H[mask_b] = (60 * ((R[mask_b] - G[mask_b]) / diff[mask_b]) + 240)

        H = (H / 2).astype(np.uint8)
        S = (S * 255).astype(np.uint8)
        V = (V * 255).astype(np.uint8)

        return np.stack([H, S, V], axis=2)
    
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
    
    # @staticmethod
    # def preprocess_pipeline(image_path):
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         raise ValueError(f"Cannot read image: {image_path}")

    #     image = ImagePreprocessor.resize_image(image)
        
    #     hsv = ImagePreprocessor.rgb_to_hsv(image)
    #     gray = ImagePreprocessor.rgb_to_gray(image)
        
    #     return image, hsv, gray

    @staticmethod
    def preprocess_pipeline(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

       
        image = ImagePreprocessor.resize_image(image)

        hsv = ImagePreprocessor.rgb_to_hsv(image)     
        gray = ImagePreprocessor.rgb_to_gray(image)   

        return image, hsv, gray