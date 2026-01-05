import os
import time
import cv2
from datetime import datetime
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

def generate_safe_filename(original_filename):
    """
    Hàm tạo tên file an toàn MỚI:
    Thay vì cố giữ tên gốc (dễ lỗi font/tiếng Việt), hàm này tạo tên mới
    dựa trên thời gian thực + đuôi file gốc.
    Ví dụ: 'Ảnh Đi Biển.jpg' -> 'img_20240101_120000_123456.jpg'
    """
    # Lấy đuôi file (extension)
    if '.' in original_filename:
        ext = original_filename.rsplit('.', 1)[1].lower()
    else:
        ext = 'jpg' # Mặc định
        
    # Tạo chuỗi thời gian duy nhất
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    return f"img_{timestamp}.{ext}"

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
    """Upload nhiều ảnh vào thư viện"""
    if 'files' not in request.files:
        return redirect(url_for('library'))
    
    files = request.files.getlist('files')
    uploaded_count = 0
    failed_files = []
    
    print(f"\n{'='*60}")
    print(f"📤 Starting upload: {len(files)} files")
    
    for idx, file in enumerate(files, 1):
        # Bỏ qua check tên file quá khắt khe, chỉ cần có file
        if not file or file.filename == '':
            continue
            
        try:
            # Dùng hàm tạo tên file an toàn mới
            safe_filename = generate_safe_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
            
            print(f"📥 [{idx}/{len(files)}] Saving: {file.filename} -> {safe_filename}")
            
            # Lưu file
            file.save(filepath)
            
            # Kiểm tra file có đọc được không bằng OpenCV
            # (Bước này quan trọng để đảm bảo preprocessing.py không bị lỗi sau này)
            test_img = cv2.imread(filepath)
            if test_img is None:
                print(f"   ❌ Cannot read image with cv2.imread()")
                os.remove(filepath)  # Xóa file lỗi
                failed_files.append(f"{file.filename} (corrupted/unreadable)")
                continue
            
            # Thêm vào database
            image_id = db.add_image(filepath)
            
            if image_id:
                uploaded_count += 1
                print(f"   ✓ Added to database: {image_id}")
            else:
                failed_files.append(f"{file.filename} (db error)")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            failed_files.append(f"{file.filename} ({str(e)})")
    
    # Lưu database
    db.save_database()
    return redirect(url_for('library'))

@app.route('/delete/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Xóa ảnh"""
    try:
        image = db.get_image(image_id)
        if image:
            if os.path.exists(image['path']):
                os.remove(image['path'])
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
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        db.clear_database()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Trang truy vấn ảnh"""
    images = db.get_all_images()
    
    if request.method == 'POST':
        query_path = None
        
        # 1. ƯU TIÊN KIỂM TRA FILE UPLOAD (Đã sửa lỗi tiếng Việt)
        file = request.files.get('query_image')
        
        if file and file.filename:
            try:
                # Tạo tên file query an toàn
                filename = 'query_' + generate_safe_filename(file.filename)
                query_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                file.save(query_path)
            except Exception as e:
                print(f"Query upload error: {e}")
                
        # 2. NẾU KHÔNG CÓ UPLOAD, KIỂM TRA THƯ VIỆN
        elif request.form.get('image_id'):
            image_id = request.form.get('image_id')
            image_data = db.get_image(image_id)
            if image_data:
                query_path = image_data['path']
            
        if query_path and os.path.exists(query_path):
            # Thực hiện tìm kiếm
            results = db.search(query_path, top_k=Config.TOP_K_RESULTS)
            
            # Chuẩn bị đường dẫn hiển thị
            display_query = '/uploads/' + os.path.basename(query_path)
            
            return render_template('query.html', 
                                 results=results,
                                 query_image=display_query,
                                 images=images,
                                 image_count=len(images))
    
    return render_template('query.html', 
                         results=None,
                         images=images,
                         image_count=len(images))

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Trang so sánh 2 ảnh"""
    images = db.get_all_images()
    
    if request.method == 'POST':
        image_id_a = request.form.get('image_id_a')
        image_id_b = request.form.get('image_id_b')
        upload_a = request.files.get('upload_a')
        upload_b = request.files.get('upload_b')
        
        path_a = None
        path_b = None
        
        # --- XỬ LÝ ẢNH A (Đã sửa lỗi tiếng Việt) ---
        if upload_a and upload_a.filename:
            try:
                filename = 'compare_a_' + generate_safe_filename(upload_a.filename)
                path_a = os.path.join(Config.UPLOAD_FOLDER, filename)
                upload_a.save(path_a)
            except Exception as e:
                print(f"Error saving A: {e}")
        elif image_id_a:
            image_data = db.get_image(image_id_a)
            if image_data: path_a = image_data['path']
            
        # --- XỬ LÝ ẢNH B (Đã sửa lỗi tiếng Việt) ---
        if upload_b and upload_b.filename:
            try:
                filename = 'compare_b_' + generate_safe_filename(upload_b.filename)
                path_b = os.path.join(Config.UPLOAD_FOLDER, filename)
                upload_b.save(path_b)
            except Exception as e:
                print(f"Error saving B: {e}")
        elif image_id_b:
            image_data = db.get_image(image_id_b)
            if image_data: path_b = image_data['path']
        
        # So sánh
        if path_a and path_b and os.path.exists(path_a) and os.path.exists(path_b):
            comparison = SimilarityCalculator.compare_images(path_a, path_b)
            
            return render_template('compare.html', 
                                 comparison=comparison,
                                 images=images,
                                 image_count=len(images),
                                 img_a='/uploads/' + os.path.basename(path_a),
                                 img_b='/uploads/' + os.path.basename(path_b))
        else:
             return render_template('compare.html', 
                                 comparison=None,
                                 images=images,
                                 error="Vui lòng chọn đủ 2 ảnh hợp lệ",
                                 image_count=len(images))
    
    return render_template('compare.html', comparison=None, images=images, image_count=len(images))

@app.route('/cleanup', methods=['GET', 'POST'])
def cleanup():
    """Trang dọn dẹp ảnh trùng lặp"""
    duplicate_groups = None
    if request.method == 'POST':
        duplicate_groups = db.find_duplicates(threshold=Config.DUPLICATE_THRESHOLD)
    
    return render_template('cleanup.html', 
                         duplicate_groups=duplicate_groups,
                         image_count=len(db.get_all_images()))

# --- Thêm vào app.py ---

@app.route('/cleanup/auto', methods=['POST'])
def auto_cleanup():
    """Tự động xóa tất cả ảnh trùng lặp, chỉ giữ lại 1 ảnh mỗi nhóm"""
    try:
        # 1. Tìm lại các nhóm trùng lặp để đảm bảo dữ liệu mới nhất
        duplicate_groups = db.find_duplicates(threshold=Config.DUPLICATE_THRESHOLD)
        
        deleted_count = 0
        errors = []

        # 2. Duyệt qua từng nhóm
        for group in duplicate_groups:
            # group['images'] là danh sách ID: [id_giữ, id_xóa_1, id_xóa_2, ...]
            # Chúng ta giữ lại phần tử đầu tiên (index 0), xóa từ index 1 trở đi
            images_to_delete = group['images'][1:]
            
            for image_id in images_to_delete:
                image_data = db.get_image(image_id)
                if image_data:
                    try:
                        # Xóa file vật lý
                        if os.path.exists(image_data['path']):
                            os.remove(image_data['path'])
                        
                        # Xóa khỏi database
                        db.remove_image(image_id)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"Lỗi xóa {image_id}: {str(e)}")
        
        # 3. Lưu database sau khi xóa hàng loạt
        if deleted_count > 0:
            db.save_database()
            
        return jsonify({
            'success': True, 
            'deleted': deleted_count,
            'errors': errors
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/editor', methods=['GET'])
def editor():
    """Trang chỉnh sửa ảnh"""
    images = db.get_all_images()
    return render_template('editor.html', result=None, images=images, image_count=len(images))

@app.route('/editor/process', methods=['POST'])
def editor_process():
    """Xử lý ảnh trong Editor"""
    images = db.get_all_images()
    
    upload_image = request.files.get('upload_image')
    image_id = request.form.get('image_id')
    action = request.form.get('action', 'equalize')
    
    original_path = None
    
    # 1. Ưu tiên Upload (Đã sửa lỗi tiếng Việt)
    if upload_image and upload_image.filename:
        try:
            filename = 'temp_original_' + generate_safe_filename(upload_image.filename)
            original_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            upload_image.save(original_path)
        except Exception as e:
            print(f"Error saving upload: {e}")
            
    # 2. Nếu không upload, dùng Library
    elif image_id:
        image_data = db.get_image(image_id)
        if image_data:
            original_path = image_data['path']
    
    if not original_path or not os.path.exists(original_path):
        return render_template('editor.html', result=None, images=images, 
                             image_count=len(images), error="Vui lòng chọn ảnh!")
    
    try:
        # Kiểm tra OpenCV đọc được không
        image = cv2.imread(original_path)
        if image is None:
            raise ValueError("Không thể đọc file ảnh (Lỗi đường dẫn hoặc định dạng)")
            
        start_time = time.time()
        
        # Xử lý ảnh
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
            quality = action.split('_')[1]
            # Tạo tên file output an toàn
            processed_filename = 'proc_' + generate_safe_filename(os.path.basename(original_path))
            processed_path = os.path.join(Config.UPLOAD_FOLDER, processed_filename)
            compression_stats = ImageCompressor.compress_jpeg(original_path, processed_path, quality)
        
        # Lưu kết quả (nếu không phải là compress)
        if processed_image is not None:
            processed_filename = 'proc_' + generate_safe_filename(os.path.basename(original_path))
            processed_path = os.path.join(Config.UPLOAD_FOLDER, processed_filename)
            cv2.imwrite(processed_path, processed_image)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            'original_path': '/uploads/' + os.path.basename(original_path),
            'processed_path': '/uploads/' + processed_filename,
            'action': action,
            'processing_time': processing_time
        }
        
        if compression_stats:
            result.update({
                'original_size': round(compression_stats['original_size_mb'] * 1024, 2),
                'processed_size': round(compression_stats['compressed_size_mb'] * 1024, 2),
                'compression_ratio': round(compression_stats['compression_ratio'], 1)
            })
        else:
            result.update({
                'original_size': round(os.path.getsize(original_path) / 1024, 2),
                'processed_size': round(os.path.getsize(processed_path) / 1024, 2)
            })
            
        return render_template('editor.html', result=result, images=images, image_count=len(images))
                             
    except Exception as e:
        print(f"Processing error: {e}")
        return render_template('editor.html', result=None, images=images, 
                             image_count=len(images), error=f"Lỗi xử lý: {str(e)}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)

@app.route('/api/stats')
def api_stats():
    """API lấy thống kê"""
    return jsonify(db.get_statistics())

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return redirect(url_for('library'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)