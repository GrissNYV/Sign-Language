# file: app/services.py (PyTorch version)
import os
import json
import threading
from typing import Tuple, Optional

import numpy as np
import torch
import mediapipe as mp

from .torch_model import BiLSTMClassifier

# --- Khởi tạo các đối tượng/toàn cục ---
MODEL_LOADED = False
model: Optional[BiLSTMClassifier] = None
actions: list[str] = ["error"]
holistic = None
SEQUENCE_LENGTH = 60  # Giá trị mặc định (sẽ cập nhật từ meta nếu có)
FEATURES_PER_FRAME = 258  # Pose + Hands
# Thống kê chuẩn hóa (đọc từ checkpoint meta nếu có)
FEAT_MEAN: Optional[np.ndarray] = None
FEAT_STD: Optional[np.ndarray] = None

# Khóa để đảm bảo an toàn khi truy cập MediaPipe trong môi trường đa luồng/async
holistic_lock = threading.Lock()

# Thiết bị suy luận
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    MODEL_PATH = "models/action_model_best.pt"
    LABELS_PATH = "models/labels.json"

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        # Đọc nhãn
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            actions = json.load(f)

        # Nạp checkpoint PyTorch
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
            meta = state.get('meta', {})
        else:
            state_dict = state
            meta = {}

        # Đọc meta nếu có
        SEQUENCE_LENGTH = int(meta.get('sequence_length', SEQUENCE_LENGTH))
        FEATURES_PER_FRAME = int(meta.get('n_features', FEATURES_PER_FRAME))
        
        # Thử nạp thống kê chuẩn hóa
        mean_list = meta.get('feat_mean')
        std_list = meta.get('feat_std')
        if mean_list is not None and std_list is not None:
            try:
                FEAT_MEAN = np.asarray(mean_list, dtype=np.float32)
                FEAT_STD = np.asarray(std_list, dtype=np.float32)
                FEAT_STD[FEAT_STD < 1e-6] = 1e-6
                if FEAT_MEAN.shape[0] != FEATURES_PER_FRAME or FEAT_STD.shape[0] != FEATURES_PER_FRAME:
                    print("[services] Cảnh báo: kích thước mean/std không khớp n_features; bỏ qua chuẩn hóa.")
                    FEAT_MEAN, FEAT_STD = None, None
            except Exception:
                FEAT_MEAN, FEAT_STD = None, None

        # Khởi tạo model và nạp trọng số
        model = BiLSTMClassifier(n_features=FEATURES_PER_FRAME, n_classes=len(actions))
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        # Khởi tạo MediaPipe Holistic
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        MODEL_LOADED = True
        print("[services] Tải model (.pt) và MediaPipe thành công. Thiết bị:", DEVICE)
    else:
        print(f"[services] Không tìm thấy '{MODEL_PATH}' hoặc '{LABELS_PATH}'. Hãy huấn luyện và tạo checkpoint PyTorch.")

except Exception as e:
    print(f"[services] Lỗi khi tải model PyTorch: {e}. API sẽ trả về lỗi khi suy luận.")


def extract_keypoints_optimized(image_rgb: np.ndarray) -> np.ndarray:
    """Trích xuất keypoints CHỈ từ Pose và Hands (đồng bộ với build_dataset.py).

    Trả về vector 1 chiều float32 có kích thước FEATURES_PER_FRAME (258) hoặc zeros nếu chưa sẵn sàng.
    """
    if not MODEL_LOADED or holistic is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

    # Đảm bảo thread-safe khi dùng chung holistic trong nhiều request
    with holistic_lock:
        results = holistic.process(image_rgb)

    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4, dtype=np.float32)
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    return np.concatenate([pose, lh, rh])


def extract_keypoints(image_rgb: np.ndarray) -> np.ndarray:
    """Alias để tương thích ngược – trỏ tới phiên bản đã tối ưu."""
    return extract_keypoints_optimized(image_rgb)


def predict_sequence(keypoints_sequence: np.ndarray | list[list[float]] | list[np.ndarray]) -> Tuple[str, float]:
    """Dự đoán hành động từ một chuỗi keypoints bằng PyTorch.

    Input: keypoints_sequence có shape (T, FEATURES_PER_FRAME)
    Output: (label, confidence)
    """
    if not MODEL_LOADED or model is None:
        return "Model chưa được tải", 0.0

    seq = np.asarray(keypoints_sequence, dtype=np.float32)

    # Chỉ lấy đúng độ dài SEQUENCE_LENGTH từ đuôi (nếu dài hơn) hoặc giữ nguyên (nếu đã đúng)
    if seq.shape[0] > SEQUENCE_LENGTH:
        seq = seq[-SEQUENCE_LENGTH:]

    # Nếu ngắn hơn, pad zeros ở đầu cho đủ chiều
    if seq.shape[0] < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - seq.shape[0]
        pad = np.zeros((pad_len, FEATURES_PER_FRAME), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)

    # Áp dụng chuẩn hóa nếu có
    if FEAT_MEAN is not None and FEAT_STD is not None and FEAT_MEAN.shape[0] == seq.shape[1]:
        denom = np.where(FEAT_STD < 1e-6, 1e-6, FEAT_STD)
        seq = (seq - FEAT_MEAN) / denom

    # (1, T, F)
    x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    label = actions[pred_idx] if 0 <= pred_idx < len(actions) else "unknown"
    return label, confidence