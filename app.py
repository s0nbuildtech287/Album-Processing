import os
import time
import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from config import Config
from services.database import ImageDatabase
from services.preprocessing import ImagePreprocessor
from services.compression import ImageCompressor
from services.similarity import SimilarityCalculator

# Khởi tạo Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app()

# Khởi tạo database
db = ImageDatabase()

def allowed_file(filename):
    """Kiểm tra file có phải ảnh hợp lệ không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# ==================== ROUTES ====================

@app.route('/')
def library():
    """Trang thư viện ảnh"""
    images = db.get_all_images()
    return render_template('library.html', 
                         images=images, 
                         image_count=len(images))

@app.route('/upload', methods=['POST'])
def upload():
    """Upload nhiều ảnh - FAST MODE"""
    if 'files' not in request.files:
        return redirect(url_for('library'))
    
    files = request.files.getlist('files')
    uploaded_count = 0
    failed_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
                
                # Lưu file
                file.save(filepath)
                
                # Thêm vào database (có xử lý lỗi)
                image_id = db.add_image(filepath)
                if image_id:
                    uploaded_count += 1
                else:
                    failed_files.append(filename)
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
                failed_files.append(filename)
    
    # Lưu database
    db.save_database()
    
    if failed_files:
        print(f"Failed to upload: {', '.join(failed_files)}")
    
    return redirect(url_for('library'))

@app.route('/delete/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Xóa ảnh"""
    try:
        # Lấy thông tin ảnh
        image = db.get_image(image_id)
        if image:
            # Xóa file vật lý
            if os.path.exists(image['path']):
                os.remove(image['path'])
            
            # Xóa khỏi database
            db.remove_image(image_id)
            db.save_database()
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Image not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear', methods=['POST'])
def clear():
    """Xóa toàn bộ database"""
    try:
        # Xóa tất cả file
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        
        # Xóa database
        db.clear_database()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Trang truy vấn ảnh"""
    if request.method == 'POST':
        if 'query_image' not in request.files:
            return redirect(url_for('query'))
        
        file = request.files['query_image']
        if file and allowed_file(file.filename):
            # Lưu ảnh query tạm
            filename = 'query_' + secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Tìm kiếm
            results = db.search(filepath, top_k=Config.TOP_K_RESULTS)
            
            # Xóa file tạm
            os.remove(filepath)
            
            return render_template('query.html', 
                                 results=results,
                                 image_count=len(db.get_all_images()))
    
    return render_template('query.html', 
                         results=None,
                         image_count=len(db.get_all_images()))

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Trang so sánh 2 ảnh"""
    if request.method == 'POST':
        if 'image_a' not in request.files or 'image_b' not in request.files:
            return redirect(url_for('compare'))
        
        file_a = request.files['image_a']
        file_b = request.files['image_b']
        
        if file_a and file_b and allowed_file(file_a.filename) and allowed_file(file_b.filename):
            # Lưu ảnh tạm
            filename_a = 'compare_a_' + secure_filename(file_a.filename)
            filename_b = 'compare_b_' + secure_filename(file_b.filename)
            
            filepath_a = os.path.join(Config.UPLOAD_FOLDER, filename_a)
            filepath_b = os.path.join(Config.UPLOAD_FOLDER, filename_b)
            
            file_a.save(filepath_a)
            file_b.save(filepath_b)
            
            # So sánh
            comparison = SimilarityCalculator.compare_images(filepath_a, filepath_b)
            
            # Xóa file tạm
            os.remove(filepath_a)
            os.remove(filepath_b)
            
            return render_template('compare.html', 
                                 comparison=comparison,
                                 image_count=len(db.get_all_images()))
    
    return render_template('compare.html', 
                         comparison=None,
                         image_count=len(db.get_all_images()))

@app.route('/cleanup', methods=['GET', 'POST'])
def cleanup():
    """Trang dọn dẹp ảnh trùng lặp"""
    duplicate_groups = None
    
    if request.method == 'POST':
        # Tìm ảnh trùng lặp
        duplicate_groups = db.find_duplicates(threshold=Config.DUPLICATE_THRESHOLD)
    
    return render_template('cleanup.html', 
                         duplicate_groups=duplicate_groups,
                         image_count=len(db.get_all_images()))

@app.route('/editor', methods=['GET'])
def editor():
    """Trang chỉnh sửa ảnh"""
    return render_template('editor.html',
                         result=None,
                         image_count=len(db.get_all_images()))

@app.route('/editor/process', methods=['POST'])
def editor_process():
    """Xử lý ảnh"""
    if 'image' not in request.files:
        return redirect(url_for('editor'))
    
    file = request.files['image']
    action = request.form.get('action', 'equalize')
    
    if file and allowed_file(file.filename):
        # Lưu ảnh gốc
        original_filename = 'original_' + secure_filename(file.filename)
        original_path = os.path.join(Config.UPLOAD_FOLDER, original_filename)
        file.save(original_path)
        
        # Đọc ảnh
        image = cv2.imread(original_path)
        start_time = time.time()
        
        # Xử lý theo action
        processed_image = None
        compression_stats = None
        
        if action == 'equalize':
            processed_image = ImagePreprocessor.equalize_histogram(image)
        elif action == 'contrast':
            processed_image = ImagePreprocessor.enhance_contrast(image, alpha=1.5)
        elif action == 'denoise_gaussian':
            processed_image = ImagePreprocessor.reduce_noise(image, method='gaussian')
        elif action == 'denoise_median':
            processed_image = ImagePreprocessor.reduce_noise(image, method='median')
        elif action == 'denoise_bilateral':
            processed_image = ImagePreprocessor.reduce_noise(image, method='bilateral')
        elif action == 'sharpen':
            processed_image = ImagePreprocessor.sharpen_image(image)
        elif action == 'binary':
            processed_image = ImagePreprocessor.convert_to_binary(image)
        elif action.startswith('compress_'):
            quality = action.split('_')[1]  # high, medium, low
            processed_filename = 'compressed_' + secure_filename(file.filename)
            processed_path = os.path.join(Config.UPLOAD_FOLDER, processed_filename)
            
            compression_stats = ImageCompressor.compress_jpeg(
                original_path, 
                processed_path, 
                quality
            )
        
        # Lưu ảnh đã xử lý (nếu không phải nén)
        if processed_image is not None:
            processed_filename = 'processed_' + secure_filename(file.filename)
            processed_path = os.path.join(Config.UPLOAD_FOLDER, processed_filename)
            cv2.imwrite(processed_path, processed_image)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Chuẩn bị kết quả
        result = {
            'original_path': '/uploads/' + original_filename,
            'processed_path': '/uploads/' + processed_filename,
            'action': action,
            'processing_time': processing_time
        }
        
        # Thêm thông tin nén nếu có
        if compression_stats:
            result['original_size'] = round(compression_stats['original_size_mb'] * 1024, 2)
            result['processed_size'] = round(compression_stats['compressed_size_mb'] * 1024, 2)
            result['compression_ratio'] = round(compression_stats['compression_ratio'], 1)
        else:
            result['original_size'] = round(os.path.getsize(original_path) / 1024, 2)
            result['processed_size'] = round(os.path.getsize(processed_path) / 1024, 2)
        
        return render_template('editor.html',
                             result=result,
                             image_count=len(db.get_all_images()))
    
    return redirect(url_for('editor'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)

@app.route('/api/stats')
def api_stats():
    """API lấy thống kê"""
    stats = db.get_statistics()
    return jsonify(stats)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return redirect(url_for('library'))

@app.errorhandler(500)
def internal_error(error):
    return f"<h1>500 Internal Server Error</h1><p>{error}</p>", 500

# ==================== MAIN ====================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)