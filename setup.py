"""
Script kiểm tra và setup môi trường CBIR
Chạy script này trước khi chạy app.py
"""

import os
import sys

def check_structure():
    """Kiểm tra cấu trúc thư mục"""
    print("🔍 Kiểm tra cấu trúc thư mục...")
    
    required_dirs = [
        'services',
        'templates',
        'uploads',
        'static'
    ]
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  ✅ Đã tạo thư mục: {dir_name}")
        else:
            print(f"  ✓ Thư mục tồn tại: {dir_name}")
    
    # Check services files
    required_services = [
        'services/__init__.py',
        'services/preprocessing.py',
        'services/compression.py',
        'services/lbp.py',
        'services/features.py',
        'services/similarity.py',
        'services/database.py'
    ]
    
    missing = []
    for file in required_services:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("\n⚠️  Thiếu các file:")
        for f in missing:
            print(f"  ❌ {f}")
    else:
        print("\n✅ Tất cả service files đều tồn tại")
    
    # Check templates
    required_templates = [
        'templates/base.html',
        'templates/library.html',
        'templates/query.html',
        'templates/compare.html',
        'templates/cleanup.html',
        'templates/editor.html'
    ]
    
    missing_templates = []
    for template in required_templates:
        if not os.path.exists(template):
            missing_templates.append(template)
    
    if missing_templates:
        print("\n⚠️  Thiếu templates:")
        for t in missing_templates:
            print(f"  ❌ {t}")
    else:
        print("✅ Tất cả templates đều tồn tại")

def check_dependencies():
    """Kiểm tra dependencies"""
    print("\n🔍 Kiểm tra dependencies...")
    
    required = [
        'cv2',
        'numpy',
        'sklearn',
        'skimage',
        'flask',
        'PIL'
    ]
    
    missing = []
    for module in required:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            missing.append(module)
            print(f"  ❌ {module}")
    
    if missing:
        print("\n⚠️  Cần cài đặt:")
        print("  pip install -r requirements.txt")
    else:
        print("\n✅ Tất cả dependencies đã được cài đặt")

def create_init_files():
    """Tạo __init__.py files"""
    print("\n🔍 Tạo __init__.py files...")
    
    init_files = [
        'services/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Auto-generated\n')
            print(f"  ✅ Đã tạo: {init_file}")
        else:
            print(f"  ✓ Đã tồn tại: {init_file}")

def check_config():
    """Kiểm tra config.py"""
    print("\n🔍 Kiểm tra config.py...")
    
    if not os.path.exists('config.py'):
        print("  ❌ Thiếu config.py")
        return False
    
    # Try to import
    try:
        from config import Config
        print(f"  ✓ Config.MAX_IMAGE_SIZE = {Config.MAX_IMAGE_SIZE}")
        print(f"  ✓ Config.LBP_POINTS = {Config.LBP_POINTS}")
        print(f"  ✓ Config.TOP_K_RESULTS = {Config.TOP_K_RESULTS}")
        print("  ✅ config.py hợp lệ")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi import config: {e}")
        return False

def test_features():
    """Test trích xuất đặc trưng"""
    print("\n🔍 Test trích xuất đặc trưng...")
    
    try:
        from services.features import FeatureExtractor
        import numpy as np
        
        # Create dummy image
        dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite('test_image.jpg', dummy)
        
        extractor = FeatureExtractor()
        features = extractor.extract_all_features('test_image.jpg')
        
        print(f"  ✓ Color features: {features['color'].shape}")
        print(f"  ✓ Texture features: {features['texture'].shape}")
        print(f"  ✓ Shape features: {features['shape'].shape}")
        
        # Cleanup
        os.remove('test_image.jpg')
        
        print("  ✅ Feature extraction hoạt động tốt")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

def main():
    print("="*60)
    print("🚀 CBIR System Setup & Check")
    print("="*60)
    
    check_structure()
    create_init_files()
    check_dependencies()
    check_config()
    test_features()
    
    print("\n" + "="*60)
    print("✅ Setup hoàn tất!")
    print("="*60)
    print("\n📝 Các bước tiếp theo:")
    print("  1. Chạy: python app.py")
    print("  2. Truy cập: http://localhost:5000")
    print("  3. Upload ảnh và test các chức năng")
    print("\n💡 Tips:")
    print("  - Upload 5-10 ảnh để test nhanh")
    print("  - Nếu chậm, giảm MAX_IMAGE_SIZE trong config.py")
    print("  - Đọc OPTIMIZATION_GUIDE.md để tối ưu thêm")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()