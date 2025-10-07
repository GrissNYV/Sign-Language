import os
import json
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from app.torch_model import BiLSTMClassifier  # giữ nguyên kiến trúc của bạn


# --- HÀM CHÍNH ---
def main():
    # --- CẤU HÌNH ---
    SEED = 42
    DATA_DIR = 'processed_data'
    MODELS_DIR = 'models'
    MODEL_PATH = os.path.join(MODELS_DIR, 'action_model_best.pt')
    LABELS_PATH = os.path.join(MODELS_DIR, 'labels.json')
    LOG_PATH = os.path.join(MODELS_DIR, 'train_log.csv')
    REPORT_PATH = os.path.join(MODELS_DIR, 'evaluation_report.json')
    CM_PATH = os.path.join(MODELS_DIR, 'confusion_matrix.png')

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Thiết bị sử dụng: {DEVICE} ---")

    # --- 1. TẢI DỮ LIỆU ---
    print("--- Bước 1: Tải Dữ liệu ---")
    X = np.load(os.path.join(DATA_DIR, 'X.npy')).astype(np.float32)  # (N, T, F)
    y_labels = np.load(os.path.join(DATA_DIR, 'y.npy')).astype(np.int64)  # (N,)

    with open('dataset_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    actions = config['actions']
    num_classes = len(actions)

    print(f"Tải dữ liệu thành công: {X.shape[0]} mẫu, {num_classes} hành động.")
    print("\nPhân bổ dữ liệu:")
    print(Counter(y_labels))

    # --- 2. PHÂN CHIA DỮ LIỆU ---
    print("\n--- Bước 2: Phân chia Dữ liệu ---")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_labels, test_size=0.15, random_state=SEED, stratify=y_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=SEED, stratify=y_train_val
    )
    print("Chia dữ liệu hoàn tất:")
    print(f" - Tập Train: {len(X_train)} mẫu")
    print(f" - Tập Validation: {len(X_val)} mẫu")
    print(f" - Tập Test: {len(X_test)} mẫu")

    # --- 2.1 CHUẨN HÓA THEO ĐẶC TRƯNG (Z-score) ---
    flat_train = X_train.reshape(-1, X_train.shape[2])  # (N_train*T, F)
    feat_mean = flat_train.mean(axis=0)
    feat_std = flat_train.std(axis=0)
    feat_std[feat_std < 1e-6] = 1e-6

    def normalize(data):
        return (data - feat_mean) / feat_std

    X_train = normalize(X_train)
    X_val   = normalize(X_val)
    X_test  = normalize(X_test)

    n_features = X_train.shape[2]
    sequence_length = X_train.shape[1]

    # --- 2.2 DATALOADER + SAMPLER CÂN BẰNG ---
    BATCH_SIZE = 32

    # WeightedRandomSampler để cân bằng lớp trong tập train
    class_counts = Counter(y_train)
    sample_weights = np.array([1.0 / class_counts[c] for c in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    # LƯU Ý: num_workers=0 để tránh lỗi spawn trên Windows. Tăng sau nếu cần.
    NUM_WORKERS = 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    # --- 3. MODEL ---
    model = BiLSTMClassifier(n_features=n_features, n_classes=num_classes).to(DEVICE)

    # --- 4. LOSS + OPTIMIZER + SCHEDULER + AMP ---
    criterion = nn.CrossEntropyLoss()  # KHÔNG dùng class_weight khi đã có sampler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    MAX_NORM = 1.0  # gradient clipping
    WARMUP_EPOCHS = 5

    # --- 5. HUẤN LUYỆN MODEL ---
    print("\n--- Bước 4: Bắt đầu Huấn luyện Model (PyTorch) ---")
    EPOCHS = 250
    PATIENCE = 30

    best_val_acc = 0.0
    patience_counter = 0

    # CSV log header
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('epoch,loss,val_loss,acc,val_acc,lr\n')

    def set_warmup_lr(epoch):
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = epoch / max(WARMUP_EPOCHS, 1)
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4 * max(warmup_factor, 0.1)

    for epoch in range(1, EPOCHS + 1):
        set_warmup_lr(epoch)

        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss /= max(total, 1)
        train_acc = correct / max(total, 1)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)

        lr_current = optimizer.param_groups[0]['lr']
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.6f},{val_acc:.6f},{lr_current:.8f}\n")

        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"acc {train_acc:.4f} | val_acc {val_acc:.4f} | lr {lr_current:.2e}")

        # Checkpoint & EarlyStopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            state = {
                'state_dict': model.state_dict(),
                'meta': {
                    'n_features': int(n_features),
                    'sequence_length': int(sequence_length),
                    'num_classes': int(num_classes),
                    'seed': int(SEED),
                    'feat_mean': feat_mean.astype(float).tolist(),
                    'feat_std': feat_std.astype(float).tolist(),
                },
            }
            torch.save(state, MODEL_PATH)
            print("→ Đã lưu model tốt nhất.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("EarlyStopping kích hoạt.")
                break

    # --- 6. ĐÁNH GIÁ TRÊN TEST ---
    print("\n--- Bước 5: Đánh giá Model trên Tập Test ---")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    print("\n===== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST =====")
    print(f"Số mẫu test: {len(all_targets)}")
    print(f"Số nhãn: {num_classes}")
    print(f"Độ chính xác: {accuracy * 100:.2f}%")

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Ma trận Nhầm lẫn trên Tập Test')
    plt.ylabel('Nhãn Thực tế')
    plt.xlabel('Nhãn Dự đoán')
    plt.tight_layout()
    plt.savefig(CM_PATH)
    print(f"Đã lưu Ma trận Nhầm lẫn tại: {CM_PATH}")

    mis_list = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                mis_list.append((i, j, cm[i, j]))
    mis_list = sorted(mis_list, key=lambda x: x[2], reverse=True)[:5]
    top_confusions = [{"true": actions[i], "pred": actions[j], "count": int(c)} for i, j, c in mis_list]

    if top_confusions:
        print("Top nhầm lẫn nhiều nhất:")
        for item in top_confusions:
            print(f"  - Thực tế: {item['true']} | Dự đoán: {item['pred']} | Số lần: {item['count']}")
    else:
        print("Không có nhầm lẫn đáng kể.")
    print("==========================================\n")

    # --- 7. LƯU KẾT QUẢ ---
    print("\n--- Bước 6: Lưu Kết quả và Báo cáo ---")
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu danh sách nhãn tại: {LABELS_PATH}")

    report = {
        'test_accuracy': float(accuracy),
        'confusion_matrix_path': CM_PATH,
        'model_path': MODEL_PATH,
        'training_log_path': LOG_PATH,
        'device': str(DEVICE),
        'top_confusions': top_confusions,
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu báo cáo đánh giá tại: {REPORT_PATH}")

    print("\nQUÁ TRÌNH HÒAN TẤT")


# --- ENTRY POINT (bắt buộc cho Windows khi dùng DataLoader) ---
if __name__ == "__main__":
    # Đảm bảo dùng start method 'spawn' trên Windows
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()