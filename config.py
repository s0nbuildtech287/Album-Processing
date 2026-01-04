import os

class Config:
    """Cấu hình hệ thống CBIR"""
    
    # Thư mục
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
    
    # Cấu hình ảnh
    MAX_IMAGE_SIZE = (256, 256)  # Giảm kích thước để xử lý nhanh hơn
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Cấu hình trích xuất đặc trưng (TỐI ƯU)
    HSV_BINS = (6, 8, 3)  # Giảm bins để nhanh hơn
    LBP_RADIUS = 1  # Giảm radius
    LBP_POINTS = 8  # Giảm points (từ 24 -> 8)
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (16, 16)  # Tăng cell size để nhanh hơn
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # Trọng số tính similarity
    WEIGHT_COLOR = 0.4  # α
    WEIGHT_TEXTURE = 0.3  # β
    WEIGHT_SHAPE = 0.3  # γ
    
    # Cấu hình truy vấn
    TOP_K_RESULTS = 10
    DUPLICATE_THRESHOLD = 0.95  # Ngưỡng phát hiện trùng lặp
    
    # Cấu hình nén ảnh
    COMPRESSION_QUALITY = {
        'high': 95,
        'medium': 75,
        'low': 50
    }
    
    # Flask config
    SECRET_KEY = 'your-secret-key-here'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    @staticmethod
    def init_app():
        """Khởi tạo các thư mục cần thiết"""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.STATIC_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMPLATE_FOLDER, exist_ok=True)