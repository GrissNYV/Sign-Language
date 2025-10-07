# file: build_dataset.py (Tối ưu hóa - Chỉ lấy Pose và Hands)
import os
# Giảm log TensorFlow/TFLite/ABSL và tắt oneDNN để tránh thông báo
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 0=ALL,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault('GLOG_minloglevel', '2')      # Giảm WARNING từ glog/absl C++
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0') # Tắt oneDNN custom ops
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
import cv2
import numpy as np
import mediapipe as mp
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Hàm trích xuất keypoints, mỗi process sẽ khởi tạo holistic riêng
def extract_keypoints_optimized(image_rgb, holistic):
    results = holistic.process(image_rgb)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    return np.concatenate([pose, lh, rh])

# Hàm xử lý từng video (dùng cho multiprocessing)
def process_video(args):
    video_path, seq_len = args
    cap = cv2.VideoCapture(video_path)
    frames = []
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, 
                                    min_detection_confidence=0.5, 
                                    min_tracking_confidence=0.5)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = extract_keypoints_optimized(frame_rgb, holistic)
            frames.append(keypoints)
    except Exception:
        cap.release()
        holistic.close()
        return None
    cap.release()
    holistic.close()
    if len(frames) < 10:
        return None
    if len(frames) > seq_len:
        indices = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
        seq = [frames[i] for i in indices]
    else:
        pad = [np.zeros_like(frames[0]) for _ in range(seq_len - len(frames))]
        seq = frames + pad
    return np.array(seq, dtype=np.float32)

def create_dataset(config):
    """Tạo bộ dữ liệu từ video."""
    X, y = [], []
    actions = config['actions']
    seq_len = config['sequence_length']
    video_root = config['video_root']

    tasks = []
    label_indices = []
    for label_idx, action in enumerate(actions):
        action_path = os.path.join(video_root, action)
        if not os.path.isdir(action_path):
            print(f"[Cảnh báo] Bỏ qua hành động '{action}' vì không tìm thấy thư mục.")
            continue
        video_files = [f for f in os.listdir(action_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        print(f"\nĐang xử lý hành động '{action}'...")
        for video_file in video_files:
            video_path = os.path.join(action_path, video_file)
            tasks.append((video_path, seq_len))
            label_indices.append(label_idx)

    print(f"Tổng số video cần xử lý: {len(tasks)}")
    results = []
    with Pool(processes=cpu_count()) as pool:
        for res in tqdm(pool.imap(process_video, tasks), total=len(tasks)):
            results.append(res)

    # Lọc kết quả hợp lệ
    for i, seq in enumerate(results):
        if seq is not None:
            X.append(seq)
            y.append(label_indices[i])

    if not X:
        print("\n[LỖI] Không có dữ liệu nào được tạo. Vui lòng kiểm tra lại cấu hình.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("\n----------------------------------------------------")
    print("Hoàn tất quá trình tạo dữ liệu!")
    print(f"Tổng số mẫu hợp lệ: {len(X)}")
    print(f"Shape của X: {X.shape}") # Shape bây giờ sẽ là (số mẫu, 60, 258)
    print(f"Shape của y: {y.shape}")

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'y.npy'), y)
    print(f"Đã lưu dữ liệu vào thư mục: '{save_dir}'")
    print("----------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Tạo bộ dữ liệu keypoints từ video cho nhận diện hành động.")
    parser.add_argument(
        'config',
        nargs='?',
        default='dataset_config.json',
        type=str,
        help='Đường dẫn tới file cấu hình JSON (mặc định: dataset_config.json).',
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file cấu hình: '{args.config}'.")
        print("Cách dùng:")
        print("  python build_dataset.py dataset_config.json")
        print("Hoặc đặt file 'dataset_config.json' cạnh script và chạy:")
        print("  python build_dataset.py")
        return
    
    create_dataset(config)

if __name__ == '__main__':
    main()

