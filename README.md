<<<<<<< HEAD
# Sign Language Recognition Project

## 1. Cấu trúc thư mục chi tiết

```
workspace/
│
├── app/
│   ├── __init__.py                # Đánh dấu app là package Python (nên tạo nếu chưa có)
│   ├── services.py                # Chứa hàm xử lý model, trích xuất keypoints, dự đoán hành động
│   └── static/
│       └── index.html             # Giao diện web tĩnh (có thể thêm CSS, JS)
│
├── data/
│   └── <action>/                  # Thư mục cho từng hành động, chứa video gốc (.mp4, .mkv, ...)
│
├── processed_data/
│   └── <action>/                  # Thư mục cho từng hành động, chứa file đặc trưng .npy
│   ├── X.npy                      # Dữ liệu đặc trưng toàn bộ (numpy array)
│   └── y.npy                      # Nhãn tương ứng cho X.npy
│
├── models/
│   ├── action_model.h5            # File mô hình đã huấn luyện (Keras/TensorFlow)
│   └── labels.json                # Danh sách nhãn (actions) của mô hình
│
├── archive/
│   └── nslt_2000.json             # (Dữ liệu gốc hoặc metadata, nếu có)
│
├── app.py                         # API FastAPI chính, nhận request và trả kết quả dự đoán
├── build_dataset.py               # Script chuyển video thành đặc trưng keypoints (.npy)
├── train_model.py                 # Script huấn luyện mô hình nhận diện động tác
├── organize_data.py               # Script sắp xếp/copy video từ nguồn lớn về đúng cấu trúc data/
├── gemini_services.py             # Gọi API Gemini để giải thích ý nghĩa động tác
├── dataset_config.json            # File cấu hình cho quá trình build dataset
├── requirements.txt               # Danh sách các package cần cài đặt
├── README.md                      # Tài liệu hướng dẫn và mô tả dự án
```

## 2. Chức năng các file chính

- **app/services.py**  
  Tải model, trích xuất keypoints từ ảnh, dự đoán hành động từ chuỗi keypoints.

- **app/gemini_services.py**  
  Gọi API Gemini để lấy giải thích động tác ngôn ngữ ký hiệu.

- **app/static/index.html**  
  Giao diện web tĩnh (có thể thêm CSS, JS).

- **build_dataset.py**  
  Đọc video từ `data/`, trích xuất keypoints bằng MediaPipe, lưu thành `.npy` cho từng hành động.

- **train_model.py**  
  Đọc dữ liệu đặc trưng từ `processed_data/`, huấn luyện mô hình LSTM/BiLSTM, lưu model và nhãn.

- **app.py**  
  Khởi tạo FastAPI, cung cấp các endpoint `/predict` (dự đoán động tác từ frames), `/explain` (giải thích động tác qua Gemini), và phục vụ giao diện web.

- **organize_data.py**  
  Sắp xếp/copy video từ nguồn lớn (Kaggle, WLASL, ...) về đúng cấu trúc thư mục `data/<action>/`.

- **dataset_config.json**  
  Cấu hình cho quá trình build dataset: danh sách hành động, độ dài chuỗi, đường dẫn dữ liệu.

- **requirements.txt**  
  Danh sách các package Python cần thiết cho dự án.

## 3. Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu
- Sử dụng `organize_data.py` để copy/sắp xếp video vào thư mục `data/<action>/`.
- Chỉnh sửa `dataset_config.json` để khớp với danh sách hành động bạn muốn nhận diện.

### Bước 2: Tạo đặc trưng keypoints từ video
```sh
python build_dataset.py dataset_config.json
```
- Kết quả: Tạo các file `.npy` trong `processed_data/`.

### Bước 3: Huấn luyện mô hình
```sh
python train_model.py
```
- Kết quả: Tạo file mô hình `models/action_model.h5` và nhãn `models/labels.json`.

### Bước 4: Khởi động API FastAPI
```sh
uvicorn app:app --reload
```
- Truy cập giao diện web tại [http://localhost:8000/](http://localhost:8000/) hoặc endpoint `/static`.

### Bước 5: Dự đoán và giải thích động tác
- Gửi request POST tới `/predict` với chuỗi frames (base64).
- Gửi request POST tới `/explain` với từ cần giải thích.

### Bước 6: Cài đặt môi trường
- Tạo và kích hoạt môi trường ảo:
  ```sh
  python -m venv venv
  .\venv\Scripts\activate ( C:/Users/nghia/Documents/workspace/.venv/Scripts/Activate.ps1 ) ( conda activate tf-gpu )
  ```
- Cài đặt các package:
  ```sh
  pip install -r requirements.txt
  ```

---

**Lưu ý:**  
- Đảm bảo đã cài đặt Python 3.11 và các package trong `requirements.txt`.
- Nếu gặp lỗi import, hãy tạo file `__init__.py` trong thư mục `app/`.
- Đọc kỹ log khi chạy các script để xử lý lỗi dữ liệu hoặc thiếu file.

---
Bạn có thể bổ sung chi tiết cho từng file hoặc quy trình theo nhu cầu thực tế!
=======
# Sign-Language
>>>>>>> fd1814681fb65b006492108b831c135f1dc5598f
