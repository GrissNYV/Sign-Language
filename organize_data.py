import os
import json
import shutil
from tqdm import tqdm

# --- CẤU HÌNH ---
# 1. Đường dẫn đến thư mục chứa video bạn đã tải từ Kaggle.
KAGGLE_VIDEO_DIR = "WLASL_2000"

# 2. Tên file JSON chứa thông tin (metadata) của bộ dữ liệu.
JSON_METADATA_FILE = "WLASL_v0.3.json"

# 3. Thư mục đầu ra, nơi sẽ chứa các thư mục hành động đã được sắp xếp.
OUTPUT_DATA_DIR = "data"

# 4. DANH SÁCH 40 HÀNH ĐỘNG BẠN ĐÃ CHỌN
TARGET_ACTIONS = [
    # Giao tiếp cơ bản
    "hello",
    "bye",
    "thank you",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "good",
    "i",
    "you",
    "she",
    "we",
    "they",
    "me",
    "my",
    "what",
    "where",
    "when",
    "why",
    "how",
    "who",
    "eat",
    "drink",
    "go",
    "work",
    "like",
    "want",
    "need",
    "play",
    "read",
    "write",
    "book",
    "computer",
    "home",
    "school",
    "water"
]
# --- KẾT THÚC CẤU HÌNH ---

def organize_videos():
    """
    Đọc file JSON, tìm các video tương ứng với TARGET_ACTIONS và sao chép
    chúng vào các thư mục hành động riêng biệt trong OUTPUT_DATA_DIR.
    """
    print("Bắt đầu quá trình tổ chức lại dữ liệu với bộ từ vựng mở rộng...")
    print(f"Sử dụng file metadata: {JSON_METADATA_FILE}")

    # Xóa thư mục data cũ để đảm bảo dữ liệu mới hoàn toàn
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"Đang xóa thư mục '{OUTPUT_DATA_DIR}' cũ...")
        shutil.rmtree(OUTPUT_DATA_DIR)
        
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    for action in TARGET_ACTIONS:
        safe_action_name = "".join([c for c in action if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, safe_action_name), exist_ok=True)

    try:
        with open(JSON_METADATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file '{JSON_METADATA_FILE}'. Vui lòng đặt nó vào đúng thư mục dự án.")
        return

    videos_copied_count = 0
    action_video_counts = {action: 0 for action in TARGET_ACTIONS}

    for entry in tqdm(data, desc="Đang quét file JSON"):
        gloss = entry['gloss']
        
        if gloss in TARGET_ACTIONS:
            for instance in entry['instances']:
                video_id = instance['video_id']
                source_video_path = os.path.join(KAGGLE_VIDEO_DIR, f"{video_id}.mp4")
                
                if os.path.exists(source_video_path):
                    safe_gloss_name = "".join([c for c in gloss if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
                    destination_path = os.path.join(OUTPUT_DATA_DIR, safe_gloss_name, f"{video_id}.mp4")
                    
                    if not os.path.exists(destination_path):
                        shutil.copy2(source_video_path, destination_path)
                        videos_copied_count += 1
                        action_video_counts[gloss] += 1

    print("\n----------------------------------------------------")
    print("Hoàn tất quá trình tổ chức dữ liệu!")
    print(f"Đã sao chép thành công tổng cộng {videos_copied_count} videos mới.")
    print(f"Dữ liệu của bạn đã sẵn sàng trong thư mục: '{OUTPUT_DATA_DIR}'")
    print("\nSố lượng video cho mỗi hành động:")
    for action, count in action_video_counts.items():
        print(f"- {action}: {count} videos")
    print("----------------------------------------------------")

if __name__ == '__main__':
    organize_videos()

