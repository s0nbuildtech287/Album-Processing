import os

class Config:
    """Cấu hình hệ thống CBIR - TỐI ƯU CHO XE HƠI"""
    
    # Thư mục
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    IMAGES_FOLDER = os.path.join(BASE_DIR, 'uploads', 'images')
    QUERIES_FOLDER = os.path.join(BASE_DIR, 'uploads', 'queries')
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
    
    # Cấu hình ảnh
    MAX_IMAGE_SIZE = (256, 256)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Cấu hình trích xuất đặc trưng
    HSV_BINS = (8, 12, 3)
    LBP_RADIUS = 1
    LBP_POINTS = 8
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (16, 16)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # --- TRỌNG SỐ TỐI ƯU (Ưu tiên SHAPE + TEXTURE) ---
    WEIGHT_COLOR = 0.10     # Giảm xuống 10% (tránh nhiễu màu nền)
    WEIGHT_TEXTURE = 0.25   # 25% (kết cấu bề mặt)
    WEIGHT_SHAPE = 0.55     # TĂNG lên 55% (hình dáng quan trọng nhất)
    WEIGHT_SPATIAL = 0.10   # 10% (phân bố không gian)
    
    # Cấu hình truy vấn
    TOP_K_RESULTS = 10
    DUPLICATE_THRESHOLD = 0.92
    
    # --- FILTERING CÂN BẰNG (Đủ kết quả + Chính xác) ---
    MIN_CONFIDENCE = 0.50        # Giảm từ 0.60 → 0.50 (nới lỏng)
    MIN_SHAPE_SIMILARITY = 0.40  # Giảm từ 0.50 → 0.40 (nới lỏng)
    MIN_COMBINED_SCORE = 0.40    # Giảm từ 0.50 → 0.40 (nới lỏng)
    MIN_TEXTURE_SIMILARITY = 0.25  # Giảm từ 0.30 → 0.25 (nới lỏng)
    
    # --- EDGE-BASED FILTERING (MỚI) ---
    MIN_EDGE_SIMILARITY = 0.35   # Edge pattern phải giống ít nhất 35%
    
    # Foreground detection
    FOREGROUND_SALIENCY_THRESHOLD = 100
    MIN_FOREGROUND_RATIO = 0.10
    MAX_FOREGROUND_RATIO = 0.95
    
    # Cấu hình nén ảnh
    COMPRESSION_QUALITY = {
        'high': 95,
        'medium': 75,
        'low': 50
    }
    
    SECRET_KEY = 'your-secret-key-here'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024

    @staticmethod
    def init_app():
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.IMAGES_FOLDER, exist_ok=True)
        os.makedirs(Config.QUERIES_FOLDER, exist_ok=True)
        os.makedirs(Config.STATIC_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMPLATE_FOLDER, exist_ok=True)