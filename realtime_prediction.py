# file: realtime_prediction.py (PyTorch version, GPU ready)
import os
import json
import cv2
import numpy as np
import torch
import mediapipe as mp

from app.torch_model import BiLSTMClassifier

# --- TẢI MÔ HÌNH VÀ CÁC THIẾT LẬP ---
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Model sẽ chạy trên: {DEVICE}")

    MODEL_PATH = "models/action_model_best.pt"
    LABELS_PATH = "models/labels.json"

    # Nhãn
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

    SEQUENCE_LENGTH = int(meta.get('sequence_length', 60))
    n_features = int(meta.get('n_features', 258))

    # Thống kê chuẩn hóa (nếu có)
    FEAT_MEAN = None
    FEAT_STD = None
    mean_list = meta.get('feat_mean')
    std_list = meta.get('feat_std')
    if mean_list is not None and std_list is not None:
        try:
            FEAT_MEAN = np.asarray(mean_list, dtype=np.float32)
            FEAT_STD = np.asarray(std_list, dtype=np.float32)
            FEAT_STD[FEAT_STD < 1e-6] = 1e-6
            if FEAT_MEAN.shape[0] != n_features or FEAT_STD.shape[0] != n_features:
                print("[INFO] Cảnh báo: mean/std không khớp n_features; bỏ qua chuẩn hóa.")
                FEAT_MEAN, FEAT_STD = None, None
        except Exception:
            FEAT_MEAN, FEAT_STD = None, None

    model = BiLSTMClassifier(n_features=n_features, n_classes=len(actions))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    MODEL_LOADED = True
except Exception as e:
    print(f"Lỗi: Không thể tải model PyTorch. Vui lòng chạy 'train_model.py' để huấn luyện và sinh .pt. Chi tiết: {e}")
    MODEL_LOADED = False
    SEQUENCE_LENGTH = 60
    actions = ["error"]
    FEAT_MEAN = None
    FEAT_STD = None

THRESHOLD = 0.7  # ngưỡng tin cậy

# --- MediaPipe & Hàm trích xuất ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_optimized(image_rgb, holistic_model):
    results = holistic_model.process(image_rgb)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
                    dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark],
                  dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark],
                  dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    return np.concatenate([pose, lh, rh])

# --- LOGIC NHẬN DIỆN THỜI GIAN THỰC ---
def main():
    if not MODEL_LOADED:
        return

    sequence = []
    sentence = []

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = extract_keypoints_optimized(image_rgb, holistic)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                seq = np.asarray(sequence, dtype=np.float32)
                # Áp dụng chuẩn hóa nếu có
                if 'FEAT_MEAN' in globals() and 'FEAT_STD' in globals() and FEAT_MEAN is not None and FEAT_STD is not None and FEAT_MEAN.shape[0] == seq.shape[1]:
                    denom = np.where(FEAT_STD < 1e-6, 1e-6, FEAT_STD)
                    seq = (seq - FEAT_MEAN) / denom
                x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, T, F)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=-1)[0]
                    confidence, pred_idx = torch.max(probs, dim=0)
                    confidence = float(confidence.item())
                    pred_idx = int(pred_idx.item())

                if confidence > THRESHOLD:
                    predicted_action = actions[pred_idx]
                    if not sentence or predicted_action != sentence[-1]:
                        sentence.append(predicted_action)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # Hiển thị kết quả
            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Real-time Sign Language Recognition', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()