import numpy as np
from config import Config

class SimilarityCalculator:
    """Tính độ tương tự NÂNG CẤP - Multi-metric + Spatial + Smart Filtering"""
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Cosine Similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def correlation_coefficient(vec1, vec2):
        """Hệ số tương quan Pearson - Tốt cho Shape"""
        if len(vec1) < 2:
            return 0.0
        
        mean1 = np.mean(vec1)
        mean2 = np.mean(vec2)
        
        numerator = np.sum((vec1 - mean1) * (vec2 - mean2))
        denominator = np.sqrt(np.sum((vec1 - mean1)**2) * np.sum((vec2 - mean2)**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def histogram_intersection(hist1, hist2):
        """Histogram Intersection - TỐT cho Color"""
        return np.sum(np.minimum(hist1, hist2)) / (np.sum(hist1) + 1e-10)
    
    @staticmethod
    def bhattacharyya_coefficient(hist1, hist2):
        """Bhattacharyya Coefficient - TỐT cho Texture"""
        hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
        return np.sum(np.sqrt(hist1_norm * hist2_norm))
    
    @staticmethod
    def adaptive_weights(features1, features2):
        """TRỌNG SỐ ĐỘNG - Tối ưu cho xe hơi"""
        # Tính độ "phong phú" của từng feature type
        color_var = (np.var(features1['color']) + np.var(features2['color'])) / 2
        texture_var = (np.var(features1['texture']) + np.var(features2['texture'])) / 2
        shape_var = (np.var(features1['shape']) + np.var(features2['shape'])) / 2
        spatial_var = (np.var(features1.get('spatial', np.zeros(1))) + 
                      np.var(features2.get('spatial', np.zeros(1)))) / 2
        
        # Normalize
        total = color_var + texture_var + shape_var + spatial_var + 1e-10
        
        # ADAPTIVE WEIGHTS - Ưu tiên SHAPE cao hơn
        w_color = 0.05 + 0.15 * (color_var / total)      # 5-20%  (giảm xuống)
        w_texture = 0.20 + 0.15 * (texture_var / total)  # 20-35%
        w_shape = 0.50 + 0.30 * (shape_var / total)      # 50-80% (tăng lên)
        w_spatial = 0.05 + 0.10 * (spatial_var / total)  # 5-15%
        
        # Normalize để tổng = 1.0
        total_w = w_color + w_texture + w_shape + w_spatial
        
        return (w_color/total_w, w_texture/total_w, 
                w_shape/total_w, w_spatial/total_w)
    
    @staticmethod
    def compare_features(features1, features2, use_adaptive=True):
        """
        So sánh NÂNG CẤP với Spatial features
        """
        # Multi-metric approach
        color_sim = SimilarityCalculator.histogram_intersection(
            features1['color'], features2['color']
        )
        
        texture_sim = SimilarityCalculator.bhattacharyya_coefficient(
            features1['texture'], features2['texture']
        )
        
        shape_sim = SimilarityCalculator.correlation_coefficient(
            features1['shape'], features2['shape']
        )
        
        # Spatial similarity (nếu có)
        if 'spatial' in features1 and 'spatial' in features2:
            spatial_sim = SimilarityCalculator.cosine_similarity(
                features1['spatial'], features2['spatial']
            )
        else:
            spatial_sim = 0.0
        
        # Adaptive weights
        if use_adaptive:
            w_c, w_t, w_s, w_sp = SimilarityCalculator.adaptive_weights(
                features1, features2
            )
        else:
            w_c = Config.WEIGHT_COLOR
            w_t = Config.WEIGHT_TEXTURE
            w_s = Config.WEIGHT_SHAPE
            w_sp = Config.WEIGHT_SPATIAL
        
        # Combined score
        combined_score = (
            w_c * color_sim +
            w_t * texture_sim +
            w_s * shape_sim +
            w_sp * spatial_sim
        )
        
        # Confidence score (đo độ đồng thuận)
        metrics = [color_sim, texture_sim, shape_sim, spatial_sim]
        variance = np.var(metrics)
        confidence = 1.0 - min(variance * 2, 0.5)
        
        return {
            'color_similarity': float(color_sim),
            'texture_similarity': float(texture_sim),
            'shape_similarity': float(shape_sim),
            'spatial_similarity': float(spatial_sim),
            'combined_similarity': float(combined_score),
            'confidence': float(confidence),
            'weights': {
                'color': float(w_c),
                'texture': float(w_t),
                'shape': float(w_s),
                'spatial': float(w_sp)
            }
        }
    
    @staticmethod
    def smart_filter(similarities, min_confidence=None, min_shape_sim=None):
        """LỌC THÔNG MINH - NÂNG CẤP với Multi-criteria"""
        if min_confidence is None:
            min_confidence = Config.MIN_CONFIDENCE
        if min_shape_sim is None:
            min_shape_sim = Config.MIN_SHAPE_SIMILARITY
        
        filtered = []
        
        for sim in similarities:
            # Rule 1: Confidence check
            if sim.get('confidence', 1.0) < min_confidence:
                continue
            
            # Rule 2: Shape check (QUAN TRỌNG NHẤT - CHẶT HƠN)
            if sim['shape_similarity'] < min_shape_sim:
                continue
            
            # Rule 3: Texture check (Loại bỏ ảnh có texture hoàn toàn khác)
            if sim['texture_similarity'] < Config.MIN_TEXTURE_SIMILARITY:
                continue
            
            # Rule 4: Combined score check
            if sim['combined_similarity'] < Config.MIN_COMBINED_SCORE:
                continue
            
            # Rule 5: Edge similarity check (MỚI - quan trọng cho xe hơi)
            # Extract edge similarity từ shape features
            # Shape features = HOG + Edge, nên nếu shape cao thì edge cũng cao
            if sim['shape_similarity'] < Config.MIN_EDGE_SIMILARITY:
                continue
            
            filtered.append(sim)
        
        return filtered
    
    @staticmethod
    def rerank_by_consistency(similarities):
        """RE-RANKING - Ưu tiên Shape + Texture, nhưng KHÔNG loại bỏ quá nhiều"""
        for sim in similarities:
            metrics = [
                sim['color_similarity'],
                sim['texture_similarity'],
                sim['shape_similarity'],
                sim.get('spatial_similarity', 0.0)
            ]
            
            mean_metric = np.mean(metrics)
            std_metric = np.std(metrics)
            
            # Consistency score
            consistency = 1.0 / (1.0 + std_metric)
            
            # BOOST mạnh nếu shape + texture đều cao
            shape_boost = 1.0 + 0.5 * sim['shape_similarity']  # Tăng boost
            texture_boost = 1.0 + 0.3 * sim['texture_similarity']
            
            # PENALTY nhẹ nếu chỉ color cao mà shape thấp
            if sim['color_similarity'] > 0.7 and sim['shape_similarity'] < 0.35:
                color_penalty = 0.8  # Penalty nhẹ hơn (20%)
            else:
                color_penalty = 1.0
            
            # Final score
            sim['consistency'] = float(consistency)
            sim['final_score'] = (sim['combined_similarity'] * 
                                 (0.5 + 0.5 * consistency) * 
                                 shape_boost * 
                                 texture_boost *
                                 color_penalty)
        
        # Re-sort
        similarities.sort(key=lambda x: x['final_score'], reverse=True)
        
        return similarities
    
    @staticmethod
    def find_similar_images(query_features, database_features, top_k=Config.TOP_K_RESULTS):
        """TÌM ẢNH TƯƠNG TỰ - PIPELINE HOÀN CHỈNH"""
        # VALIDATE query_features
        if query_features is None:
            print("❌ Query features is None!")
            return []
        
        similarities = []
        
        # Step 1: Calculate similarities
        for image_id, db_features in database_features.items():
            # SKIP nếu db_features bị None
            if db_features is None:
                print(f"⚠️  Skipping {image_id} - features is None")
                continue
            
            try:
                sim = SimilarityCalculator.compare_features(
                    query_features, db_features, use_adaptive=True
                )
                sim['image_id'] = image_id
                similarities.append(sim)
            except Exception as e:
                print(f"⚠️  Error comparing {image_id}: {e}")
                continue
        
        # Step 2: Smart filtering
        similarities = SimilarityCalculator.smart_filter(similarities)
        
        # Step 3: Re-ranking
        similarities = SimilarityCalculator.rerank_by_consistency(similarities)
        
        # Step 4: Return top-K
        return similarities[:top_k]
    
    @staticmethod
    def compare_images(image_path1, image_path2):
        """So sánh 2 ảnh"""
        from services.features import FeatureExtractor
        
        extractor = FeatureExtractor()
        features1 = extractor.extract_all_features(image_path1)
        features2 = extractor.extract_all_features(image_path2)
        
        return SimilarityCalculator.compare_features(features1, features2)
    
    @staticmethod
    def find_duplicates(database_features, threshold=Config.DUPLICATE_THRESHOLD):
        """Phát hiện ảnh trùng lặp"""
        image_ids = list(database_features.keys())
        n = len(image_ids)
        
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = SimilarityCalculator.compare_features(
                    database_features[image_ids[i]],
                    database_features[image_ids[j]]
                )
                similarity_matrix[i, j] = sim['combined_similarity']
                similarity_matrix[j, i] = sim['combined_similarity']
        
        visited = set()
        duplicate_groups = []
        
        for i in range(n):
            if i in visited:
                continue
            
            group = [image_ids[i]]
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    group.append(image_ids[j])
                    visited.add(j)
            
            if len(group) >= 2:
                duplicate_groups.append({
                    'images': group,
                    'size': len(group)
                })
            
            visited.add(i)
        
        return duplicate_groups