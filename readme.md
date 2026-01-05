# Hệ thống Truy vấn Ảnh dựa Nội dung (CBIR System)

## 📋 Tổng quan

Hệ thống CBIR (Content-Based Image Retrieval) cho phép tìm kiếm và quản lý kho ảnh dựa trên nội dung thị giác, sử dụng các kỹ thuật trích xuất đặc trưng tiên tiến.

## 🚀 Cài đặt & Chạy

### 1. Clone repository

```bash
git clone <repository-url>
cd "Album Processing"
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

## 📁 Cấu trúc Thư mục

```
Album Processing/
├── app.py                 # Flask application chính
├── config.py             # Cấu hình hệ thống
├── requirements.txt      # Dependencies
├── services/            # Business logic
│   ├── database.py      # Quản lý database ảnh
│   ├── features.py      # Trích xuất đặc trưng
│   ├── preprocessing.py # Tiền xử lý ảnh
│   ├── compression.py   # Nén ảnh
│   └── similarity.py    # Tính toán độ tương đồng
├── templates/           # Giao diện HTML
├── uploads/            # Thư mục lưu ảnh
│   ├── images/         # Ảnh thư viện (album)
│   └── queries/        # Ảnh truy vấn tạm
└── static/             # CSS, JS, assets
```

## ✨ Tính năng chính

### 1. **Quản lý Thư viện Theo Album**

- 📁 Tổ chức ảnh theo album/thư mục (Động vật, Đồ vật, Con người...)
- ✏️ Đổi tên album dễ dàng
- 📤 Upload nhiều ảnh cùng lúc với tên tùy chỉnh
- 🗑️ Xóa ảnh đơn lẻ hoặc xóa toàn bộ database

### 2. **Truy vấn Ảnh Tương tự**

- 🔍 Tìm top-K ảnh tương tự nhất với ảnh mẫu
- 📊 Hiển thị điểm tương đồng chi tiết:
  - Màu sắc (HSV Histogram)
  - Kết cấu (Local Binary Pattern - LBP)
  - Hình dạng (Histogram of Oriented Gradients - HOG)
- 🎯 Kết quả được sắp xếp theo độ tương đồng tổng hợp

### 3. **So sánh 2 Ảnh**

- 📸 Upload 2 ảnh bất kỳ để so sánh
- 📈 Hiển thị điểm tương đồng chi tiết cho từng đặc trưng
- 📊 Trực quan hóa bằng biểu đồ thanh

### 4. **Dọn dẹp Ảnh Trùng lặp**

- 🔎 Tự động phát hiện nhóm ảnh trùng/gần trùng
- 📑 Hiển thị theo nhóm để dễ quản lý
- 🗑️ Xóa từng ảnh hoặc xóa cả nhóm

### 5. **Chỉnh sửa & Cải thiện Ảnh**

- 🎨 Cân bằng Histogram: Cải thiện độ tương phản
- ✨ Tăng độ tương phản: Làm rõ chi tiết
- 🔇 Giảm nhiễu: 3 phương pháp (Gaussian, Median, Bilateral)
- 🔪 Làm sắc nét: Tăng độ sắc nét
- ⚫⚪ Chuyển đen trắng: Binary threshold
- 📦 Nén ảnh: 3 mức chất lượng (High/Medium/Low)

## 🔧 Cấu trúc Database

Database sử dụng **relative paths** để đảm bảo tính portable:

```python
{
  "image_id": {
    "path": "uploads/images/conmeo.jpg",  # Relative path
    "album": "Động vật",                   # Album name
    "metadata": {...},                     # Thông tin ảnh
    "features": {...}                      # Đặc trưng (lazy load)
  }
}
```

### Migration Tự động

- ✅ Tự động convert absolute paths cũ → relative paths
- ✅ Tự động thêm field `album` cho ảnh cũ
- ✅ Đảm bảo backward compatibility

## 🎯 Workflow Sử dụng

### Upload ảnh vào album:

1. Nhấn "Tải ảnh lên"
2. Chọn file ảnh
3. Popup hiện ra:
   - Nhập tên album (ví dụ: "Động vật", "Đồ vật")
   - Nhập tên từng ảnh (hoặc để trống)
4. Nhấn "Upload"

### Xem ảnh theo album:

1. Màn hình chính hiển thị các folder 📁
2. Click vào folder để xem ảnh bên trong
3. Click breadcrumb hoặc nút "←" để quay lại

### Đổi tên album:

1. Click nút ✏️ ở góc album
2. Nhập tên mới
3. Tất cả ảnh trong album được cập nhật

## 🔍 Cấu trúc dự án

│ ├── preprocessing.py # Tiền xử lý ảnh
│ ├── compression.py # Nén ảnh
│ ├── lbp.py # LBP (TỰ CÀI ĐẶT)
│ ├── features.py # Trích xuất đặc trưng
│ ├── similarity.py # Tính độ tương tự
│ └── database.py # Quản lý database
├── templates/
│ ├── base.html # Base template
│ ├── library.html # Trang thư viện
│ ├── query.html # Trang truy vấn
│ ├── compare.html # Trang so sánh
│ ├── cleanup.html # Trang dọn dẹp
│ └── editor.html # Trang chỉnh sửa
├── uploads/ # Thư mục lưu ảnh
├── image_database.pkl # Database đặc trưng
└── metadata.json # Metadata ảnh
🚀 Cài đặt

1. Cài đặt Python packages
   bashpip install -r requirements.txt
2. Chạy ứng dụng
   bashpython app.py
3. Truy cập
   Mở trình duyệt và truy cập: http://localhost:5000
   🔬 Chi tiết kỹ thuật
   Pipeline xử lý ảnh

Resize về kích thước chuẩn (512x512)
Chuyển đổi không gian màu:

RGB → HSV (cho đặc trưng màu)
RGB → Grayscale (cho kết cấu và hình dạng)

Trích xuất đặc trưng

1. Đặc trưng Màu sắc (HSV Histogram)

Sử dụng histogram trên không gian màu HSV
Bins: (8, 12, 3) cho H, S, V
Chuẩn hóa histogram

2. Đặc trưng Kết cấu (LBP - TỰ CÀI ĐẶT)

Local Binary Pattern được tự cài đặt hoàn toàn
Radius: 3, Points: 24
Sử dụng uniform patterns
Nội suy song tuyến tính (Bilinear Interpolation)
Histogram chuẩn hóa

Cách hoạt động:
python# Với mỗi pixel trung tâm:

# 1. Lấy 24 điểm xung quanh trên vòng tròn bán kính 3

# 2. So sánh giá trị với pixel trung tâm

# 3. Tạo binary code 24-bit

# 4. Xây dựng histogram

3. Đặc trưng Hình dạng (HOG)

Histogram of Oriented Gradients
Orientations: 9
Pixels per cell: (8, 8)
Cells per block: (2, 2)
Chuẩn hóa L2

Tính độ tương tự
Cosine Similarity
similarity = (A · B) / (||A|| _ ||B||)
Điểm tổng hợp
score = α _ color*sim + β * texture*sim + γ * shape_sim
Với trọng số mặc định:

α (Color) = 0.4
β (Texture) = 0.3
γ (Shape) = 0.3

Nén ảnh
Hỗ trợ 3 mức chất lượng:

High: JPEG quality = 95
Medium: JPEG quality = 75
Low: JPEG quality = 50

📊 Cấu hình
File config.py chứa các tham số quan trọng:
python# Kích thước ảnh
MAX_IMAGE_SIZE = (512, 512)

# Histogram HSV

HSV_BINS = (8, 12, 3)

# LBP parameters

LBP_RADIUS = 3
LBP_POINTS = 24

# HOG parameters

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)

# Trọng số similarity

WEIGHT_COLOR = 0.4
WEIGHT_TEXTURE = 0.3
WEIGHT_SHAPE = 0.3

# Top-K results

TOP_K_RESULTS = 10

# Ngưỡng duplicate

DUPLICATE_THRESHOLD = 0.95
🎯 Sử dụng

1. Thư viện Ảnh

Click "Tải ảnh lên" để upload nhiều ảnh
Hệ thống tự động trích xuất đặc trưng
Xem danh sách ảnh dạng lưới

2. Truy vấn Ảnh

Chọn ảnh mẫu để tìm kiếm
Hệ thống trả về top-10 ảnh tương tự
Xem điểm chi tiết cho từng loại đặc trưng

3. So sánh 2 Ảnh

Upload 2 ảnh cần so sánh
Click "Phân tích So sánh"
Xem kết quả với visualization

4. Dọn dẹp

Click "Quét trùng lặp"
Hệ thống tìm các nhóm ảnh tương tự
Xóa theo nhóm hoặc từng ảnh

5. Chỉnh sửa Ảnh

Chọn ảnh và chức năng xử lý
Click "Áp dụng"
So sánh kết quả trước/sau
Tải xuống ảnh đã xử lý

🔧 API Endpoints
GET / # Thư viện
POST /upload # Upload ảnh
DELETE /delete/<id> # Xóa ảnh
POST /clear # Xóa database

GET /query # Trang truy vấn
POST /query # Thực hiện truy vấn

GET /compare # Trang so sánh
POST /compare # So sánh 2 ảnh

GET /cleanup # Trang dọn dẹp
POST /cleanup # Tìm trùng lặp

GET /editor # Trang chỉnh sửa
POST /editor/process # Xử lý ảnh

GET /api/stats # Lấy thống kê
📝 Lưu ý

Database: Được lưu dưới dạng pickle file (image_database.pkl)
Metadata: Được lưu riêng dưới dạng JSON để dễ đọc
Upload folder: Chứa tất cả ảnh gốc và ảnh đã xử lý
Performance: Với >1000 ảnh, nên tối ưu bằng cách sử dụng index tree (KD-Tree, Ball Tree)

🎨 Giao diện

Thiết kế hiện đại với Tailwind CSS
Responsive design (mobile-friendly)
Dark/light mode support
Smooth animations và transitions
Loading indicators

📚 Tài liệu tham khảo

LBP: Ojala et al. (2002) - Multiresolution Gray-Scale and Rotation Invariant Texture Classification
HOG: Dalal & Triggs (2005) - Histograms of Oriented Gradients for Human Detection
CBIR: Smeulders et al. (2000) - Content-Based Image Retrieval at the End of the Early Years

👨‍💻 Tác giả
Dự án bài tập lớn môn Dữ liệu Đa phương tiện
📄 License
MIT License
