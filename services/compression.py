import cv2
import os
from config import Config

class ImageCompressor:
    """Nén ảnh để giảm dung lượng"""
    
    @staticmethod
    def compress_jpeg(input_path, output_path, quality='medium'):
        """
        Nén ảnh dạng JPEG
        quality: 'high' (95), 'medium' (75), 'low' (50)
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")
        
        q = Config.COMPRESSION_QUALITY.get(quality, 75)
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, q])
        
        return ImageCompressor.get_compression_stats(input_path, output_path)
    
    @staticmethod
    def compress_png(input_path, output_path, compression_level=6):
        """
        Nén ảnh dạng PNG
        compression_level: 0-9 (0: no compression, 9: max compression)
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")
        
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        
        return ImageCompressor.get_compression_stats(input_path, output_path)
    
    @staticmethod
    def compress_webp(input_path, output_path, quality=80):
        """
        Nén ảnh dạng WebP
        quality: 1-100
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")
        
        cv2.imwrite(output_path, image, [cv2.IMWRITE_WEBP_QUALITY, quality])
        
        return ImageCompressor.get_compression_stats(input_path, output_path)
    
    @staticmethod
    def get_compression_stats(original_path, compressed_path):
        """Thống kê kết quả nén"""
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        
        ratio = (1 - compressed_size / original_size) * 100
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': ratio,
            'size_saved': original_size - compressed_size,
            'size_saved_mb': (original_size - compressed_size) / (1024 * 1024)
        }
    
    @staticmethod
    def batch_compress(input_folder, output_folder, format='jpeg', quality='medium'):
        """
        Nén hàng loạt ảnh trong thư mục
        """
        os.makedirs(output_folder, exist_ok=True)
        
        results = []
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(tuple(Config.ALLOWED_EXTENSIONS)):
                continue
            
            input_path = os.path.join(input_folder, filename)
            
            # Đổi extension theo format
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.{format}"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                if format == 'jpeg' or format == 'jpg':
                    stats = ImageCompressor.compress_jpeg(input_path, output_path, quality)
                elif format == 'png':
                    level = 3 if quality == 'low' else 6 if quality == 'medium' else 9
                    stats = ImageCompressor.compress_png(input_path, output_path, level)
                elif format == 'webp':
                    q = 50 if quality == 'low' else 80 if quality == 'medium' else 95
                    stats = ImageCompressor.compress_webp(input_path, output_path, q)
                else:
                    continue
                
                stats['filename'] = filename
                results.append(stats)
            except Exception as e:
                print(f"Error compressing {filename}: {e}")
        
        return results