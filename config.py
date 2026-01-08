import os

class Config:
    """Cấu hình hệ thống CBIR"""
    
    # Thư mục
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    IMAGES_FOLDER = os.path.join(BASE_DIR, 'uploads', 'images')  
    QUERIES_FOLDER = os.path.join(BASE_DIR, 'uploads', 'queries')  
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
    
 
    MAX_IMAGE_SIZE = (256, 256)  
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    

    HSV_BINS = (6, 8, 3)  
    LBP_RADIUS = 1  
    LBP_POINTS = 8  
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (16, 16)  
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    
    USE_DEEP_FEATURES = True  
    DEEP_MODEL = 'resnet50' 
    
    WEIGHT_COLOR = 0.15     
    WEIGHT_TEXTURE = 0.15   
    WEIGHT_SHAPE = 0.15     
    WEIGHT_DEEP = 0.55      
    
    # Cấu hình truy vấn
    TOP_K_RESULTS = 4
    DUPLICATE_THRESHOLD = 0.85  
    # Cấu hình nén ảnh
    COMPRESSION_QUALITY = {
        'high': 95,
        'medium': 75,
        'low': 50
    }
    
    # Flask config
    SECRET_KEY = 'your-secret-key-here'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  
    
    @staticmethod
    def init_app():
        """Khởi tạo các thư mục cần thiết"""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.IMAGES_FOLDER, exist_ok=True)
        os.makedirs(Config.QUERIES_FOLDER, exist_ok=True)
        os.makedirs(Config.STATIC_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMPLATE_FOLDER, exist_ok=True)