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
    """Trang thư viện ảnh - hiển thị theo album"""
    # Lấy tham số album từ URL (nếu có)
    selected_album = request.args.get('album')
    
    if selected_album:
        # Hiển thị ảnh của album cụ thể
        images = db.get_images_by_album(selected_album)
        albums = db.get_albums()
        return render_template('library.html', 
                             images=images, 
                             image_count=len(images),
                             albums=albums,
                             selected_album=selected_album)
    else:
        # Hiển thị danh sách album
        albums = db.get_albums()
        return render_template('library.html', 
                             images=None,
                             image_count=0,
                             albums=albums,
                             selected_album=None)

@app.route('/upload', methods=['POST'])
def upload():
    """Upload nhiều ảnh vào thư viện với tên tùy chỉnh"""
    if 'files' not in request.files:
        return redirect(url_for('library'))
    
    files = request.files.getlist('files')
    custom_names = request.form.getlist('custom_names')  # Lấy danh sách tên tùy chỉnh
    album_name = request.form.get('album_name', 'Uncategorized').strip()  # Lấy tên album
    uploaded_count = 0
    failed_files = []
    
    print(f"\n{'='*60}")
    print(f"📤 Starting upload: {len(files)} files to album '{album_name}'")
    
    for idx, file in enumerate(files, 1):
        # Bỏ qua check tên file quá khắt khe, chỉ cần có file
        if not file or file.filename == '':
            continue
            
        try:
            # Lấy tên tùy chỉnh từ form, nếu không có thì tự động tạo
            if idx <= len(custom_names) and custom_names[idx-1].strip():
                custom_name = custom_names[idx-1].strip()
                # Lấy extension từ file gốc
                ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
                safe_filename = f"{custom_name}.{ext}"
            else:
                # Tạo tên tự động nếu không có tên tùy chỉnh
                safe_filename = generate_safe_filename(file.filename)
            
            filepath = os.path.join(Config.IMAGES_FOLDER, safe_filename)
            
            print(f"📥 [{idx}/{len(files)}] Saving: {file.filename} -> {safe_filename}")
            
            # Lưu file
            file.save(filepath)
            
            # Kiểm tra file có đọc được không bằng OpenCV
            test_img = cv2.imread(filepath)
            if test_img is None:
                print(f"   ❌ Cannot read image with cv2.imread()")
                os.remove(filepath)  # Xóa file lỗi
                failed_files.append(f"{file.filename} (corrupted/unreadable)")
                continue
            
            # Thêm vào database với album
            image_id = db.add_image(filepath, album=album_name)
            
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

@app.route('/rename_album', methods=['POST'])
def rename_album():
    """Đổi tên album"""
    try:
        old_name = request.json.get('old_name')
        new_name = request.json.get('new_name')
        
        if not old_name or not new_name:
            return jsonify({'success': False, 'error': 'Missing album name'})
        
        success = db.rename_album(old_name, new_name)
        
        if success:
            db.save_database()
            return jsonify({'success': True, 'new_name': new_name})
        else:
            return jsonify({'success': False, 'error': 'Album not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Xóa ảnh"""
    try:
        image = db.get_image(image_id)
        if image:
            # Lấy absolute path để xóa file
            absolute_path = db.get_absolute_path(image_id)
            if absolute_path and os.path.exists(absolute_path):
                os.remove(absolute_path)
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
                # Tạo tên file query an toàn và lưu vào QUERIES_FOLDER
                filename = 'query_' + generate_safe_filename(file.filename)
                query_path = os.path.join(Config.QUERIES_FOLDER, filename)
                file.save(query_path)
            except Exception as e:
                print(f"Query upload error: {e}")
                
        # 2. NẾU KHÔNG CÓ UPLOAD, KIỂM TRA THƯ VIỆN
        elif request.form.get('image_id'):
            image_id = request.form.get('image_id')
            image_data = db.get_image(image_id)
            if image_data:
                query_path = db.get_absolute_path(image_id)
            
        if query_path and os.path.exists(query_path):
            # Thực hiện tìm kiếm
            results = db.search(query_path, top_k=Config.TOP_K_RESULTS)
            
            # Chuẩn bị đường dẫn hiển thị
            # Xác định xem ảnh nằm trong thư mục nào
            if 'images' in query_path:
                display_query = '/uploads/images/' + os.path.basename(query_path)
            elif 'queries' in query_path:
                display_query = '/uploads/queries/' + os.path.basename(query_path)
            else:
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
                path_a = os.path.join(Config.QUERIES_FOLDER, filename)
                upload_a.save(path_a)
            except Exception as e:
                print(f"Error saving A: {e}")
        elif image_id_a:
            image_data = db.get_image(image_id_a)
            if image_data: path_a = db.get_absolute_path(image_id_a)
            
        # --- XỬ LÝ ẢNH B (Đã sửa lỗi tiếng Việt) ---
        if upload_b and upload_b.filename:
            try:
                filename = 'compare_b_' + generate_safe_filename(upload_b.filename)
                path_b = os.path.join(Config.QUERIES_FOLDER, filename)
                upload_b.save(path_b)
            except Exception as e:
                print(f"Error saving B: {e}")
        elif image_id_b:
            image_data = db.get_image(image_id_b)
            if image_data: path_b = db.get_absolute_path(image_id_b)
        
        # So sánh
        if path_a and path_b and os.path.exists(path_a) and os.path.exists(path_b):
            comparison = SimilarityCalculator.compare_images(path_a, path_b)
            
            # Xác định đường dẫn hiển thị
            if 'images' in path_a:
                display_a = '/uploads/images/' + os.path.basename(path_a)
            elif 'queries' in path_a:
                display_a = '/uploads/queries/' + os.path.basename(path_a)
            else:
                display_a = '/uploads/' + os.path.basename(path_a)
                
            if 'images' in path_b:
                display_b = '/uploads/images/' + os.path.basename(path_b)
            elif 'queries' in path_b:
                display_b = '/uploads/queries/' + os.path.basename(path_b)
            else:
                display_b = '/uploads/' + os.path.basename(path_b)
            
            return render_template('compare.html', 
                                 comparison=comparison,
                                 images=images,
                                 image_count=len(images),
                                 img_a=display_a,
                                 img_b=display_b)
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
                        # Xóa file vật lý - dùng absolute path
                        absolute_path = db.get_absolute_path(image_id)
                        if absolute_path and os.path.exists(absolute_path):
                            os.remove(absolute_path)
                        
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
            original_path = db.get_absolute_path(image_id)
    
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

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files từ cả 2 thư mục"""
    # Nếu filename chứa 'images/' hoặc 'queries/', serve từ thư mục tương ứng
    if filename.startswith('images/'):
        return send_from_directory(Config.IMAGES_FOLDER, filename[7:])
    elif filename.startswith('queries/'):
        return send_from_directory(Config.QUERIES_FOLDER, filename[8:])
    else:
        # Fallback: tìm kiếm trong cả 2 thư mục
        images_path = os.path.join(Config.IMAGES_FOLDER, filename)
        queries_path = os.path.join(Config.QUERIES_FOLDER, filename)
        upload_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        if os.path.exists(images_path):
            return send_from_directory(Config.IMAGES_FOLDER, filename)
        elif os.path.exists(queries_path):
            return send_from_directory(Config.QUERIES_FOLDER, filename)
        else:
            return send_from_directory(Config.UPLOAD_FOLDER, filename)

@app.route('/api/stats')
def api_stats():
    """API lấy thống kê"""
    return jsonify(db.get_statistics())

@app.route('/admin/rebuild-features')
def rebuild_features():
    """
    Force rebuild tất cả features với deep learning
    Hiển thị tiến trình chi tiết
    """
    print("\n" + "="*60)
    print("🔄 REBUILDING ALL FEATURES WITH DEEP LEARNING")
    print("="*60)
    
    images = db.get_all_images()
    total = len(images)
    
    results = {
        'total': total,
        'updated': 0,
        'failed': 0,
        'details': []
    }
    
    for idx, (image_id, image_data) in enumerate(images.items(), 1):
        try:
            abs_path = db.get_absolute_path(image_id)
            
            # Kiểm tra có deep features chưa
            has_deep = image_data.get('features') and 'deep' in image_data['features']
            status = "✓ Has deep" if has_deep else "✗ Missing deep"
            
            print(f"\n[{idx}/{total}] {image_id}")
            print(f"  Status: {status}")
            print(f"  Path: {abs_path}")
            
            # Force extract lại
            print(f"  🔍 Extracting features...")
            success = db.update_features(image_id)
            
            if success:
                results['updated'] += 1
                print(f"  ✅ Updated successfully!")
                results['details'].append({
                    'id': image_id,
                    'status': 'success',
                    'had_deep': has_deep
                })
            else:
                results['failed'] += 1
                print(f"  ❌ Failed to update!")
                results['details'].append({
                    'id': image_id,
                    'status': 'failed',
                    'had_deep': has_deep
                })
                
        except Exception as e:
            results['failed'] += 1
            print(f"  ❌ Error: {e}")
            results['details'].append({
                'id': image_id,
                'status': 'error',
                'error': str(e)
            })
    
    # Lưu database
    print("\n" + "="*60)
    print("💾 Saving database...")
    db.save_database()
    
    print("\n📊 SUMMARY:")
    print(f"  Total images: {results['total']}")
    print(f"  ✅ Updated: {results['updated']}")
    print(f"  ❌ Failed: {results['failed']}")
    print("="*60 + "\n")
    
    return jsonify(results)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return redirect(url_for('library'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)