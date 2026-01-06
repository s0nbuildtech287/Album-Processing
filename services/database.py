import os
import json
import pickle
import numpy as np
from datetime import datetime
from config import Config

# Dùng feature extractor tối ưu
from services.features import FeatureExtractor


class ImageDatabase:
    """Quản lý cơ sở dữ liệu ảnh và đặc trưng - LAZY LOADING"""
    
    def __init__(self, db_path='image_database.pkl'):
        self.db_path = os.path.join(Config.BASE_DIR, db_path)
        self.metadata_path = os.path.join(Config.BASE_DIR, 'metadata.json')
        self.images = {}  # {image_id: {'path': ..., 'features': ..., 'metadata': ...}}
        self.feature_extractor = FeatureExtractor(use_deep_features=Config.USE_DEEP_FEATURES)
        self.load_database()
    
    def add_image(self, image_path, image_id=None, album="Uncategorized"):
        """
        Thêm ảnh vào database - NHANH (không extract features)
        Returns: image_id
        """
        if image_id is None:
            image_id = self._generate_image_id(image_path)
        
        # Kiểm tra đã tồn tại chưa
        if image_id in self.images:
            print(f"Image {image_id} already exists in database")
            return image_id
        
        try:
            # Chỉ lấy metadata (rất nhanh!)
            metadata = self._extract_metadata(image_path)
            
            # Chuyển sang relative path để portable
            relative_path = os.path.relpath(image_path, Config.BASE_DIR)
            
            # Lưu vào database - KHÔNG trích xuất đặc trưng
            self.images[image_id] = {
                'path': relative_path,  # Lưu relative path thay vì absolute
                'features': None,  
                'metadata': metadata,
                'album': album,  # Thêm trường album
                'added_at': datetime.now().isoformat()
            }
            
            print(f"Added image {image_id} to database (album: {album})")
            return image_id
            
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
            return None
    
    def add_images_batch(self, image_paths):
        """
        Thêm nhiều ảnh cùng lúc - CỰC NHANH
        Returns: list of image_ids
        """
        image_ids = []
        for path in image_paths:
            image_id = self.add_image(path)
            if image_id:
                image_ids.append(image_id)
        return image_ids
    
    def remove_image(self, image_id):
        """Xóa ảnh khỏi database"""
        if image_id in self.images:
            del self.images[image_id]
            print(f"Removed image {image_id} from database")
            return True
        return False
    
    def get_image(self, image_id):
        """Lấy thông tin ảnh"""
        return self.images.get(image_id)
    
    def get_absolute_path(self, image_id):
        """Lấy absolute path của ảnh từ relative path"""
        image = self.images.get(image_id)
        if not image:
            return None
        
        relative_path = image['path']
        # Nếu đã là absolute path (database cũ), trả về luôn
        if os.path.isabs(relative_path):
            return relative_path
        
        # Convert relative path sang absolute path
        return os.path.join(Config.BASE_DIR, relative_path)
    
    def get_all_images(self):
        """Lấy tất cả ảnh"""
        return self.images
    
    def get_albums(self):
        """Lấy danh sách tất cả album với số lượng ảnh"""
        albums = {}
        for image_id, image_data in self.images.items():
            album = image_data.get('album', 'Uncategorized')
            if album not in albums:
                albums[album] = {
                    'name': album,
                    'count': 0,
                    'images': []
                }
            albums[album]['count'] += 1
            albums[album]['images'].append(image_id)
        return albums
    
    def get_images_by_album(self, album_name):
        """Lấy tất cả ảnh trong một album"""
        result = {}
        for image_id, image_data in self.images.items():
            if image_data.get('album', 'Uncategorized') == album_name:
                result[image_id] = image_data
        return result
    
    def rename_album(self, old_name, new_name):
        """Đổi tên album - cập nhật tất cả ảnh trong album"""
        if not new_name or new_name.strip() == "":
            return False
        
        new_name = new_name.strip()
        count = 0
        
        for image_id, image_data in self.images.items():
            if image_data.get('album', 'Uncategorized') == old_name:
                image_data['album'] = new_name
                count += 1
        
        if count > 0:
            print(f"Renamed album '{old_name}' to '{new_name}' ({count} images)")
            return True
        return False
    
    def get_features(self, image_id):
        """Lấy đặc trưng của ảnh - LAZY LOADING"""
        image = self.images.get(image_id)
        if not image:
            return None
        
        # Nếu chưa có features, trích xuất ngay
        if image['features'] is None:
            print(f"Extracting features for {image_id}...")
            self.update_features(image_id)
        
        return image['features']
    
    def get_all_features(self):
        """
        Lấy đặc trưng của tất cả ảnh - LAZY LOADING
        Returns: dict {image_id: features}
        """
        features_dict = {}
        total = len(self.images)
        
        for idx, (img_id, img_data) in enumerate(self.images.items(), 1):
            # Lazy load nếu chưa có
            if img_data['features'] is None:
                print(f"[{idx}/{total}] Extracting features for {img_id}...")
                self.update_features(img_id)
            
            features_dict[img_id] = img_data['features']
        
        return features_dict
    
    def update_features(self, image_id):
        """Cập nhật lại đặc trưng của ảnh"""
        if image_id not in self.images:
            return False
        
        try:
            # Lấy absolute path để đọc file
            absolute_path = self.get_absolute_path(image_id)
            
            # Extract với verbose logging
            features = self.feature_extractor.extract_all_features(absolute_path, verbose=True)
            
            self.images[image_id]['features'] = features
            self.images[image_id]['updated_at'] = datetime.now().isoformat()
            return True
        except Exception as e:
            print(f"Error updating features for {image_id}: {e}")
            return False
    
    def search(self, query_path, top_k=Config.TOP_K_RESULTS):
        """
        Tìm kiếm ảnh tương tự - ĐÂY MỚI LÀ LÚC EXTRACT!
        """
        from services.similarity import SimilarityCalculator
        
        # Trích xuất đặc trưng của query
        query_features = self.feature_extractor.extract_all_features(query_path)
        
        # Lấy đặc trưng của tất cả ảnh (sẽ tự động extract nếu chưa có)
        db_features = self.get_all_features()
        
        # Tìm ảnh tương tự
        results = SimilarityCalculator.find_similar_images(
            query_features, 
            db_features, 
            top_k
        )
        
        # Thêm thông tin đường dẫn
        for result in results:
            image_id = result['image_id']
            result['path'] = self.images[image_id]['path']
            result['metadata'] = self.images[image_id]['metadata']
        
        return results
    
    def find_duplicates(self, threshold=Config.DUPLICATE_THRESHOLD):
        """Tìm ảnh trùng lặp"""
        from services.similarity import SimilarityCalculator
        
        db_features = self.get_all_features()
        groups = SimilarityCalculator.find_duplicates(db_features, threshold)
        
        # Thêm thông tin chi tiết
        for group in groups:
            group['images_info'] = []
            for img_id in group['images']:
                group['images_info'].append({
                    'id': img_id,
                    'path': self.images[img_id]['path'],
                    'metadata': self.images[img_id]['metadata']
                })
        
        return groups
    
    def extract_all_features_now(self):
        """
        Force extract tất cả features ngay
        (Dùng cho background processing)
        """
        print("Force extracting all features...")
        count = 0
        for img_id, img_data in self.images.items():
            if img_data['features'] is None:
                self.update_features(img_id)
                count += 1
        print(f"Extracted features for {count} images")
        return count
    
    def save_database(self):
        """Lưu database ra file"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.images, f)
            print(f"Database saved to {self.db_path}")
            
            # Lưu metadata riêng (dễ đọc)
            metadata = {
                img_id: {
                    'path': img_data['path'],
                    'metadata': img_data['metadata'],
                    'album': img_data.get('album', 'Uncategorized'),
                    'added_at': img_data.get('added_at', ''),
                    'has_features': img_data['features'] is not None
                }
                for img_id, img_data in self.images.items()
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self):
        """Load database từ file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.images = pickle.load(f)
                print(f"Loaded {len(self.images)} images from database")
                
                # Migration: Convert absolute paths sang relative paths
                self._migrate_to_relative_paths()
                
            except Exception as e:
                print(f"Error loading database: {e}")
                self.images = {}
        else:
            print("No existing database found, starting fresh")
            self.images = {}
    
    def _migrate_to_relative_paths(self):
        """Convert absolute paths cũ sang relative paths"""
        migrated = 0
        for image_id, image_data in self.images.items():
            path = image_data['path']
            
            # Nếu là absolute path và nằm trong BASE_DIR
            if os.path.isabs(path) and Config.BASE_DIR in path:
                relative_path = os.path.relpath(path, Config.BASE_DIR)
                image_data['path'] = relative_path
                migrated += 1
            
            # Thêm album field nếu chưa có
            if 'album' not in image_data:
                image_data['album'] = 'Uncategorized'
        
        if migrated > 0:
            print(f"Migrated {migrated} images to relative paths")
            self.save_database()
    
    def clear_database(self):
        """Xóa toàn bộ database"""
        self.images = {}
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        print("Database cleared")
    
    def get_statistics(self):
        """Lấy thống kê database"""
        with_features = sum(1 for img in self.images.values() if img['features'] is not None)
        without_features = len(self.images) - with_features
        
        return {
            'total_images': len(self.images),
            'with_features': with_features,
            'without_features': without_features,
            'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
            'images': list(self.images.keys())
        }
    
    def _generate_image_id(self, image_path):
        """Tạo ID duy nhất cho ảnh"""
        filename = os.path.basename(image_path)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"{os.path.splitext(filename)[0]}_{timestamp}"
    
    def _extract_metadata(self, image_path):
        """Trích xuất metadata của ảnh - NHANH"""
        import cv2
        
        # Chỉ đọc để lấy thông tin cơ bản
        img = cv2.imread(image_path)
        
        return {
            'filename': os.path.basename(image_path),
            'size': os.path.getsize(image_path),
            'dimensions': f"{img.shape[1]}x{img.shape[0]}",
            'channels': img.shape[2] if len(img.shape) == 3 else 1
        }