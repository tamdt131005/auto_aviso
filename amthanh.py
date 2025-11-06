import os
import pygame
import threading
import time

os.environ["SDL_AUDIODRIVER"] = "directsound"  # fix lỗi audio device

# Biến điều khiển trạng thái phát âm
stop_flag = False

def play_alert(file_path):
    """Luồng phát âm thanh cảnh báo."""
    global stop_flag
    try:
        pygame.mixer.init()
        time.sleep(0.2)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(-1)  # -1 = lặp vô hạn
        print("🔊 Bắt đầu phát cảnh báo...")

        while not stop_flag:
            time.sleep(0.1)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        print("🛑 Dừng phát cảnh báo.")
    except Exception as e:
        print("Lỗi khi phát âm:", e)

def start_alert(file_path="./amthanh/canhbao_captra.mp3"):
    """Khởi động luồng phát âm thanh."""
    global stop_flag
    stop_flag = False
    t = threading.Thread(target=play_alert, args=(file_path,), daemon=True)
    t.start()
    return t  # trả về luồng để quản lý nếu cần

def stop_alert():
    """Dừng luồng phát âm thanh."""
    global stop_flag
    stop_flag = True

# -------------------------
# Ví dụ sử dụng
if __name__ == "__main__":
    print("🚀 Chương trình chính bắt đầu.")
    thread = start_alert("./amthanh/canhbao_captra.mp3")

    # Giả lập chương trình vẫn chạy bình thường
    for i in range(10):
        print(f"🏃 Đang chạy tác vụ {i} ...")
        time.sleep(1)
        if i == 5:
            print("⏹ Dừng cảnh báo tại i = 5")
            stop_alert()  # dừng luồng âm thanh
    print("✅ Kết thúc chương trình chính.")
