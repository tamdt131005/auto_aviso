import sys
import os
import time
import random

# Nhập các hàm từ module models
try:
    from models import (
        load_screenshot_bgr,
        click_task_title,
        click_confirm_button,
        check_btn_xn,
        check_captra,
        logger,
        _screenshot_buffer
    )
except ImportError as e:
    print(f"❌ Lỗi khi import: {e}")
    print("Vui lòng đảm bảo file models.py nằm cùng thư mục")
    sys.exit(1)

# Nhập mô-đun âm thanh (tùy chọn)
try:
    from amthanh import start_alert, stop_alert
    AUDIO_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  Mô-đun âm thanh không khả dụng - cảnh báo tắt")
    AUDIO_AVAILABLE = False
    def start_alert(): pass
    def stop_alert(): pass

# ============================================
# CẤU HÌNH
# ============================================

CONFIG = {
    'max_count': 50,                    # Tổng số nhiệm vụ cần hoàn thành
    'break_interval': 25,               # Nghỉ sau mỗi N nhiệm vụ
    'break_duration': (2, 5),           # Thời gian nghỉ (min, max)
    'captcha_timeout': 60,              # Thiời gian tối đa chờ captcha được giải (giây)
    'captcha_check_interval': 2,        # Khoảng kiểm tra captcha (giây)
    'button_wait_max': 15,              # Thời gian tố đa chờ nút xuất hiện (giây)
    'button_check_intervals': [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0],  # Các khoảng chờ tăng dần
    'page_load_delay': (3.5, 4.5),      # Chờ load trang (min, max)
    'post_captcha_delay': (1.0, 2.0),   # Delay sau khi captcha được giải
    'inter_action_delay': (0.5, 0.25),  # Delay giữa các hành động (cơ bản, biến thiên)
    'retry_delay': (0.8, 0.3),          # Delay trước khi thử lại khi thất bại
}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def smart_wait(base=0.3, variance=0.15):
    """
    Chờ thông minh có ngẫu nhiên hóa
    Trả về thời gian thực tế đã chờ
    """
    wait_time = max(0.1, base + random.uniform(-variance, variance))
    time.sleep(wait_time)
    return wait_time

def should_take_break(count, interval=25):
    """Kiểm tra xem có đến lúc nghỉ không"""
    return count > 0 and count % interval == 0

def take_smart_break():
    """Thực hiện nghỉ với thời lượng ngẫu nhiên"""
    duration = random.uniform(*CONFIG['break_duration'])
    logger.info(f"⏸️  Nghỉ trong {duration:.1f}s...")
    time.sleep(duration)
    _screenshot_buffer.invalidate()
    logger.info("▶️  Tiếp tục...")

def format_time(seconds):
    """Định dạng giây thành chuỗi dễ đọc"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ============================================
# CAPTCHA HANDLING
# ============================================

def wait_and_solve_captcha(max_wait=60, check_interval=2):
    """
    Chờ và xử lý captcha nếu xuất hiện

    Trả về:
        True nếu captcha đã được giải hoặc không có captcha
        False nếu quá thời gian chờ
    """
    logger.info("🔍 Kiểm tra captcha...")
    start_time = time.time()
    
    # Kiểm tra ban đầu
    screen = load_screenshot_bgr(force_refresh=True)
    
    if not check_captra(screen, threshold=0.5):
        logger.debug("✅ Không phát hiện captcha")
        return True
    
    # Phát hiện captcha
    logger.warning("🔒 PHÁT HIỆN CAPTCHA!")
    
    # Phát âm báo nếu có
    if AUDIO_AVAILABLE:
        start_alert()
        time.sleep(3)
        stop_alert()
    else:
        logger.info("🔔 [BEEP] Vui lòng giải captcha!")
    
    logger.info(f"⏳ Đang chờ tối đa {max_wait}s cho captcha được giải...")
    
    captcha_start = time.time()
    checks = 0
    
    while time.time() - captcha_start < max_wait:
        time.sleep(check_interval)
        checks += 1
        
        screen = load_screenshot_bgr(force_refresh=True)
        
        # Kiểm tra nếu captcha đã biến mất
        if not check_captra(screen, threshold=0.5):
            elapsed = time.time() - captcha_start
            logger.info(f"✅ Captcha đã được giải sau {elapsed:.1f}s ({checks} lần kiểm tra)")
            _screenshot_buffer.invalidate()
            return True
        
        elapsed = time.time() - captcha_start
        remaining = max_wait - elapsed
        logger.debug(f"⏳ Vẫn đang chờ... ({remaining:.0f}s còn lại, kiểm tra #{checks})")
    
    # Hết thời gian chờ
    logger.error(f"❌ Hết thời gian chờ captcha sau {max_wait}s")
    return False

# ============================================
# BUTTON WAITING
# ============================================

def wait_for_button(check_intervals=None, threshold=0.7):
    """
    Chờ nút xác nhận xuất hiện với kiểm tra tăng dần

    Trả về:
        (found, screen, wait_time) tuple
    """
    if check_intervals is None:
        check_intervals = CONFIG['button_check_intervals']
    
    logger.info("🔍 Đang chờ nút xác nhận...")
    
    total_waited = 0
    
    for idx, interval in enumerate(check_intervals):
        # Nghỉ
        time.sleep(interval)
        total_waited += interval
        
        # Chụp ảnh mới
        screen = load_screenshot_bgr(force_refresh=True)
        
        # Kiểm tra nút
        if check_btn_xn(screen_bgr=screen, threshold=threshold, debug=False):
            logger.info(f"✅ Đã tìm thấy nút sau {total_waited:.1f}s!")
            return True, screen, total_waited
        
        logger.debug(f"⏳ Chưa có... ({total_waited:.1f}s đã chờ, lần thử {idx+1}/{len(check_intervals)})")
    
    logger.warning(f"⏱️  Hết thời gian chờ nút sau {total_waited:.1f}s")
    return False, None, total_waited

# ============================================
# STATISTICS TRACKING
# ============================================

class Stats:
    def __init__(self):
        self.success_count = 0
        self.fail_count = 0
        self.captcha_count = 0
        self.start_time = time.time()
        self.button_wait_times = []
    
    def record_success(self):
        self.success_count += 1
    
    def record_failure(self):
        self.fail_count += 1
    
    def record_captcha(self):
        self.captcha_count += 1
    
    def record_button_wait(self, wait_time):
        self.button_wait_times.append(wait_time)
    
    def get_elapsed(self):
        return time.time() - self.start_time
    
    def get_avg_time(self):
        if self.success_count == 0:
            return 0
        return self.get_elapsed() / self.success_count
    
    def get_rate(self):
        elapsed_minutes = self.get_elapsed() / 60
        if elapsed_minutes == 0:
            return 0
        return self.success_count / elapsed_minutes
    
    def get_success_rate(self):
        total = self.success_count + self.fail_count
        if total == 0:
            return 0
        return (self.success_count / total) * 100
    
    def get_avg_button_wait(self):
        if not self.button_wait_times:
            return 0
        return sum(self.button_wait_times) / len(self.button_wait_times)
    
    def print_progress(self, current, target):
        elapsed = self.get_elapsed()
        avg_time = self.get_avg_time()
        remaining = avg_time * (target - current)
        rate = self.get_rate()
        
        logger.info(f"✅ Đã hoàn thành {current}/{target}")
        logger.info(f"📊 Thành công: {self.success_count} | Thất bại: {self.fail_count} | Captcha: {self.captcha_count}")
        logger.info(f"⚡ Tốc độ: {rate:.1f}/phút | Trung bình: {avg_time:.1f}s/nhiệm vụ")
        logger.info(f"🕐 Đã chạy: {format_time(elapsed)} | ETA: {format_time(remaining)}")
        
        if self.button_wait_times:
            avg_btn_wait = self.get_avg_button_wait()
            logger.info(f"⏱️  Thời gian chờ nút trung bình: {avg_btn_wait:.1f}s")
    
    def print_final(self, target):
        total_time = self.get_elapsed()
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 HOÀN THÀNH TỰ ĐỘNG HÓA!")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Thành công: {self.success_count}/{target}")
        logger.info(f"❌ Thất bại: {self.fail_count}")
        logger.info(f"🔒 Số lần gặp captcha: {self.captcha_count}")
        logger.info(f"⏱️  Tổng thời gian: {format_time(total_time)}")
        
        if self.success_count > 0:
            avg = self.get_avg_time()
            rate = self.get_rate()
            efficiency = self.get_success_rate()
            
            logger.info(f"📊 Thời gian trung bình: {avg:.2f}s mỗi nhiệm vụ")
            logger.info(f"⚡ Tốc độ: {rate:.1f} nhiệm vụ/phút")
            logger.info(f"🎯 Tỉ lệ thành công: {efficiency:.1f}%")
            
            if self.captcha_count > 0:
                captcha_rate = (self.captcha_count / target) * 100
                logger.info(f"🔒 Tỉ lệ captcha: {captcha_rate:.1f}%")
            
            if self.button_wait_times:
                avg_btn_wait = self.get_avg_button_wait()
                min_wait = min(self.button_wait_times)
                max_wait = max(self.button_wait_times)
                logger.info(f"⏱️  Thời gian đợi nút: trung bình={avg_btn_wait:.1f}s, nhỏ nhất={min_wait:.1f}s, lớn nhất={max_wait:.1f}s")
        
        logger.info(f"{'='*60}")

# ============================================
# MAIN WORKFLOW
# ============================================

def execute_single_task(stats):
    """
    Execute a single task cycle
    
    Returns:
        True if successful, False if failed
    """
    
    # ============================================
    # Step 1: Capture screen and click task
    # ============================================
    logger.info("📸 Step 1: Capture screen and click task...")
    screen = load_screenshot_bgr(use_cache=False, force_refresh=True)
    
    if not click_task_title(screen_bgr=screen, debug=False):
        logger.warning("⚠️  Task not found")
        return False
    
    logger.info("✅ Clicked task")
    
    # ============================================
    # Step 2: Wait for page load & check captcha
    # ============================================
    logger.info("⏱️  Step 2: Waiting for page load & checking captcha...")
    
    # Wait for page to load
    page_load_time = random.uniform(*CONFIG['page_load_delay'])
    time.sleep(page_load_time)
    
    # Check and solve captcha if present
    if not wait_and_solve_captcha(
        max_wait=CONFIG['captcha_timeout'],
        check_interval=CONFIG['captcha_check_interval']
    ):
        logger.error("❌ Failed to solve captcha")
        stats.record_captcha()
        return False
    
    # If captcha was present and solved, wait for UI refresh
    if stats.captcha_count > 0:
        logger.info("⏳ Waiting for UI refresh after captcha...")
        post_captcha_delay = random.uniform(*CONFIG['post_captcha_delay'])
        time.sleep(post_captcha_delay)
    
    # ============================================
    # Step 3: Wait for confirm button
    # ============================================
    logger.info("🔍 Step 3: Waiting for confirm button...")
    
    btn_found, screen, wait_time = wait_for_button(
        check_intervals=CONFIG['button_check_intervals']
    )
    
    if not btn_found:
        logger.warning("⏱️  Button timeout")
        return False
    
    stats.record_button_wait(wait_time)
    
    # Minimal delay before click
    time.sleep(random.uniform(0.05, 0.15))
    
    # ============================================
    # Step 4: Click confirm button
    # ============================================
    logger.info("👆 Step 4: Click confirm button...")
    
    if not click_confirm_button(screen_bgr=screen, debug=False):
        logger.warning("⚠️  Failed to click confirm button")
        return False
    
    logger.info("✅ Clicked confirm button")
    return True

# ============================================
# MAIN LOOP
# ============================================

def main():
    """Main execution loop"""
    
    # Configuration
    max_count = CONFIG['max_count']
    
    # Statistics
    stats = Stats()
    count = 0
    
    # Print header
    logger.info("=" * 60)
    logger.info("🚀 ULTRA SPEED AUTOMATION - WITH CAPTCHA HANDLING")
    logger.info("=" * 60)
    logger.info(f"🎯 Target: {max_count} tasks")
    logger.info(f"⚡ Optimizations:")
    logger.info(f"   - Template pre-loaded & cached with scaled versions")
    logger.info(f"   - Screenshot buffer reuse (TTL=300ms)")
    logger.info(f"   - Early exit when match score > 0.85")
    logger.info(f"   - Ultra-fast delays (10-50ms)")
    logger.info(f"   - Smart captcha detection & handling")
    logger.info(f"   - Progressive button waiting")
    logger.info("=" * 60)
    logger.info("📋 Workflow:")
    logger.info("   Click task → Check captcha → Wait button → Click confirm")
    logger.info("=" * 60)
    
    # Initial delay
    time.sleep(random.uniform(0.5, 1.0))
    
    # Main loop
    while count < max_count:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 Task [{count + 1}/{max_count}]")
            logger.info(f"{'='*50}")
            
            # Take break if needed
            if should_take_break(count, interval=CONFIG['break_interval']):
                take_smart_break()
            
            # Execute task
            success = execute_single_task(stats)
            
            if success:
                count += 1
                stats.record_success()
                stats.print_progress(count, max_count)
            else:
                stats.record_failure()
                logger.warning("❌ Task failed. Retrying...")
                smart_wait(*CONFIG['retry_delay'])
                continue
            
            # Inter-action delay
            inter_delay = smart_wait(*CONFIG['inter_action_delay'])
            logger.debug(f"⏱️  Inter-action delay: {inter_delay:.2f}s")
            
        except KeyboardInterrupt:
            logger.info("\n\n⛔ Stopped by user (Ctrl+C)")
            break
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            stats.record_failure()
            logger.info("⏳ Waiting 2s before retry...")
            smart_wait(2.0, 0.5)
            _screenshot_buffer.invalidate()
            continue
    
    # Print final statistics
    stats.print_final(max_count)

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)