import numpy as np
from config import Config

class SimilarityCalculator:
    """Tính độ tương tự giữa các ảnh"""
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Tính Cosine Similarity giữa 2 vector
        similarity = (A · B) / (||A|| * ||B||)
        """
        # Kiểm tra kích thước vector
        if vec1.shape != vec2.shape:
            # Nếu kích thước khác nhau, resize về kích thước nhỏ hơn
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            # Nếu không có dữ liệu, trả về 0
            if min_len == 0:
                return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def euclidean_distance(vec1, vec2):
        """Tính khoảng cách Euclidean"""
        return np.linalg.norm(vec1 - vec2)
    
    @staticmethod
    def manhattan_distance(vec1, vec2):
        """Tính khoảng cách Manhattan"""
        return np.sum(np.abs(vec1 - vec2))
    
    @staticmethod
    def chi_square_distance(hist1, hist2):
        """
        Tính khoảng cách Chi-Square (thường dùng cho histogram)
        """
        eps = 1e-10
        chi_square = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
        return chi_square
    
    @staticmethod
    def compare_features(features1, features2):
        """
        So sánh các đặc trưng giữa 2 ảnh
        Returns: dict with individual and combined similarities
        """
        try:
            # Kiểm tra features hợp lệ
            if not all(k in features1 for k in ['color', 'texture', 'shape']):
                return {
                    'color_similarity': 0.0,
                    'texture_similarity': 0.0,
                    'shape_similarity': 0.0,
                    'combined_similarity': 0.0
                }
            
            if not all(k in features2 for k in ['color', 'texture', 'shape']):
                return {
                    'color_similarity': 0.0,
                    'texture_similarity': 0.0,
                    'shape_similarity': 0.0,
                    'combined_similarity': 0.0
                }
            
            # Tính similarity cho từng loại đặc trưng
            color_sim = SimilarityCalculator.cosine_similarity(
                features1['color'], 
                features2['color']
            )
            
            texture_sim = SimilarityCalculator.cosine_similarity(
                features1['texture'], 
                features2['texture']
            )
            
            shape_sim = SimilarityCalculator.cosine_similarity(
                features1['shape'], 
                features2['shape']
            )
            
            # Tính điểm tổng hợp
            # score = α * color + β * texture + γ * shape
            combined_score = (
                Config.WEIGHT_COLOR * color_sim +
                Config.WEIGHT_TEXTURE * texture_sim +
                Config.WEIGHT_SHAPE * shape_sim
            )
            
            return {
                'color_similarity': float(color_sim),
                'texture_similarity': float(texture_sim),
                'shape_similarity': float(shape_sim),
                'combined_similarity': float(combined_score)
            }
        except Exception as e:
            print(f"Error comparing features: {e}")
            return {
                'color_similarity': 0.0,
                'texture_similarity': 0.0,
                'shape_similarity': 0.0,
                'combined_similarity': 0.0
            }
    
    @staticmethod
    def compare_images(image_path1, image_path2):
        """
        So sánh 2 ảnh hoàn chỉnh
        """
        from services.features import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Trích xuất đặc trưng
        features1 = extractor.extract_all_features(image_path1)
        features2 = extractor.extract_all_features(image_path2)
        
        # Tính similarity
        result = SimilarityCalculator.compare_features(features1, features2)
        
        return result
    
    @staticmethod
    def find_similar_images(query_features, database_features, top_k=Config.TOP_K_RESULTS):
        """
        Tìm top-K ảnh tương tự nhất
        query_features: đặc trưng ảnh truy vấn
        database_features: dict {image_id: features}
        """
        similarities = []
        
        for image_id, db_features in database_features.items():
            sim = SimilarityCalculator.compare_features(query_features, db_features)
            similarities.append({
                'image_id': image_id,
                **sim
            })
        
        # Sắp xếp theo combined_similarity giảm dần
        similarities.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        # Lấy top-K
        return similarities[:top_k]
    
    @staticmethod
    def find_duplicates(database_features, threshold=Config.DUPLICATE_THRESHOLD):
        """
        Phát hiện các nhóm ảnh trùng/gần trùng
        Returns: list of duplicate groups
        """
        try:
            # Lọc các ảnh có features hợp lệ
            valid_features = {}
            for img_id, features in database_features.items():
                if features and all(k in features for k in ['color', 'texture', 'shape']):
                    # Kiểm tra không phải là dummy features (all zeros)
                    if (np.any(features['color']) or 
                        np.any(features['texture']) or 
                        np.any(features['shape'])):
                        valid_features[img_id] = features
            
            image_ids = list(valid_features.keys())
            n = len(image_ids)
            
            if n < 2:
                return []  # Không đủ ảnh để so sánh
            
            # Ma trận similarity
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        sim = SimilarityCalculator.compare_features(
                            valid_features[image_ids[i]],
                            valid_features[image_ids[j]]
                        )
                        similarity_matrix[i, j] = sim['combined_similarity']
                        similarity_matrix[j, i] = sim['combined_similarity']
                    except Exception as e:
                        print(f"Error comparing {image_ids[i]} and {image_ids[j]}: {e}")
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
            
            # Tìm các nhóm trùng lặp
            visited = set()
            duplicate_groups = []
            
            for i in range(n):
                if i in visited:
                    continue
                
                # Tìm các ảnh tương tự với ảnh i
                group = [image_ids[i]]
                for j in range(i + 1, n):
                    if similarity_matrix[i, j] >= threshold:
                        group.append(image_ids[j])
                        visited.add(j)
                
                # Chỉ thêm nhóm nếu có >= 2 ảnh
                if len(group) >= 2:
                    duplicate_groups.append({
                        'images': group,
                        'size': len(group)
                    })
                
                visited.add(i)
            
            return duplicate_groups
        except Exception as e:
            print(f"Error in find_duplicates: {e}")
            return []