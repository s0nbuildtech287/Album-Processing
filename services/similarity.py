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
        So sánh các đặc trưng giữa 2 ảnh (bao gồm deep features)
        Returns: dict with individual and combined similarities
        """
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
        
        # Tính deep similarity nếu có
        deep_sim = 0.0
        if 'deep' in features1 and 'deep' in features2:
            deep_sim = SimilarityCalculator.cosine_similarity(
                features1['deep'], 
                features2['deep']
            )
        
        # Tính điểm tổng hợp
        # score = α * color + β * texture + γ * shape + δ * deep
        combined_score = (
            Config.WEIGHT_COLOR * color_sim +
            Config.WEIGHT_TEXTURE * texture_sim +
            Config.WEIGHT_SHAPE * shape_sim +
            Config.WEIGHT_DEEP * deep_sim
        )
        
        return {
            'color_similarity': float(color_sim),
            'texture_similarity': float(texture_sim),
            'shape_similarity': float(shape_sim),
            'deep_similarity': float(deep_sim),
            'combined_similarity': float(combined_score)
        }
    
    @staticmethod
    def compare_images(image_path1, image_path2):
        """
        So sánh 2 ảnh hoàn chỉnh
        """
        from services.features import FeatureExtractor
        
        extractor = FeatureExtractor(use_deep_features=Config.USE_DEEP_FEATURES)
        
        # Trích xuất đặc trưng
        features1 = extractor.extract_all_features(image_path1)
        features2 = extractor.extract_all_features(image_path2)
        
        # Tính similarity
        result = SimilarityCalculator.compare_features(features1, features2)
        
        return result
    
    @staticmethod
    def find_similar_images(query_features, database_features, top_k=Config.TOP_K_RESULTS):
        similarities = []
        
        for image_id, db_features in database_features.items():
            sim = SimilarityCalculator.compare_features(query_features, db_features)
            similarities.append({
                'image_id': image_id,
                **sim
            })

        similarities.sort(key=lambda x: x['combined_similarity'], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def find_duplicates(database_features, threshold=Config.DUPLICATE_THRESHOLD):
        """
        Phát hiện các nhóm ảnh trùng/gần trùng
        Returns: list of duplicate groups
        """
        image_ids = list(database_features.keys())
        n = len(image_ids)
        
        # Ma trận similarity
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = SimilarityCalculator.compare_features(
                    database_features[image_ids[i]],
                    database_features[image_ids[j]]
                )
                similarity_matrix[i, j] = sim['combined_similarity']
                similarity_matrix[j, i] = sim['combined_similarity']
        
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