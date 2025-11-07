import subprocess
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import io
import time
import sys
import os
import logging
import random
from concurrent.futures import ThreadPoolExecutor
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CẤU HÌNH CHUNG
# ============================================

# Scales mặc định cho tất cả template matching
DEFAULT_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Scales cho các template cụ thể (có thể tùy chỉnh)
TEMPLATE_SCALES = {
    'item_nv': [0.8, 0.9, 1.0, 1.1, 1.2],  # Tiêu đề nhiệm vụ
    'btn_xacnhan': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],  # Nút xác nhận
    'captra': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],  # Captcha
}

# ============================================
# TỐI ƯU HÓA NÂNG CAO: Bộ nhớ đệm nhiều cấp
# ============================================

class TemplateCache:
    """Cache template với các phiên bản đã được scale sẵn"""
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get(self, path, scales=None):
        """Lấy template đã scale sẵn từ cache"""
        if scales is None:
            scales = DEFAULT_SCALES
        
        cache_key = (path, tuple(scales))
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Load template
            template = cv2.imread(path)
            if template is None:
                logger.error(f"❌ Không đọc được template: {path}")
                return None
            
            # Pre-compute tất cả scales
            scaled_templates = []
            temp_h, temp_w = template.shape[:2]
            
            for scale in scales:
                if scale == 1.0:
                    scaled_templates.append((template, scale, temp_w, temp_h))
                else:
                    new_w = int(temp_w * scale)
                    new_h = int(temp_h * scale)
                    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    scaled_templates.append((resized, scale, new_w, new_h))
            
            self._cache[cache_key] = scaled_templates
            logger.info(f"✅ Đã cache template: {os.path.basename(path)} với {len(scales)} tỉ lệ")
            return scaled_templates
    
    def clear(self):
        """Xóa cache"""
        with self._lock:
            self._cache.clear()
            logger.info("🗑️  Đã xóa cache template")

class ScreenshotBuffer:
    """Buffer để tái sử dụng screenshot trong cùng 1 cycle"""
    def __init__(self, ttl=0.3):
        self._buffer = None
        self._timestamp = 0
        self._ttl = ttl
        self._lock = threading.Lock()
    
    def get(self, force_refresh=False):
        """Lấy screenshot, tái sử dụng nếu còn fresh"""
        with self._lock:
            current_time = time.time()
            
            if not force_refresh and self._buffer is not None:
                age = current_time - self._timestamp
                if age < self._ttl:
                    logger.debug(f"♻️  Tái sử dụng ảnh chụp (tuổi: {age:.2f}s)")
                    return self._buffer
            
            logger.debug("📸 Đang chụp ảnh màn hình mới")
            data = adb_screencap_bytes()
            img = Image.open(io.BytesIO(data))
            self._buffer = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self._timestamp = current_time
            
            return self._buffer
    
    def invalidate(self):
        """Vô hiệu hóa cache sau khi tap/swipe"""
        with self._lock:
            self._timestamp = 0

# Global instances
_template_cache = TemplateCache()
_screenshot_buffer = ScreenshotBuffer(ttl=0.3)

# ============================================
# HÀM CHÍNH: Tìm kiếm template đa tỉ lệ
# ============================================

def match_template_multiscale(screen_bgr, template_path, threshold=0.6, 
                              scales=None, early_exit_conf=0.9, debug=False):
    """
    🎯 HÀM CHÍNH: Tìm kiếm template đa tỉ lệ
    
    Tham số:
        screen_bgr: Ảnh màn hình dạng BGR
        template_path: Đường dẫn tới file template
        threshold: Ngưỡng độ tin cậy (0.0-1.0)
        scales: Danh sách tỉ lệ cần thử (None = dùng mặc định)
        early_exit_conf: Ngưỡng để dừng sớm khi tìm thấy match rất tốt
        debug: Nếu True lưu ảnh debug
    
    Trả về:
        dict: {
            'found': bool,
            'confidence': float,
            'location': (x, y),  # Tọa độ tâm
            'bbox': (x, y, w, h),
            'scale': float
        }
    """
    result = {
        'found': False,
        'confidence': 0.0,
        'location': None,
        'bbox': None,
        'scale': 1.0
    }
    
    # Lấy screen size
    screen_h, screen_w = screen_bgr.shape[:2]
    
    # Lấy scaled templates từ cache
    scaled_templates = _template_cache.get(template_path, scales=scales)
    if scaled_templates is None:
        return result
    
    best_val = 0
    best_match = None
    best_scale = 1.0
    
    # Thử tất cả scales
    for template, scale, temp_w, temp_h in scaled_templates:
        # Skip nếu template lớn hơn screen
        if temp_w > screen_w or temp_h > screen_h:
            logger.debug(f"⏭️  Skip scale {scale:.2f} (quá lớn: {temp_w}x{temp_h})")
            continue
        
        # Match template
        match_result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
        
        logger.debug(f"📏 Scale {scale:.2f} ({temp_w}x{temp_h}) -> conf={max_val:.4f}")
        
        # Cập nhật best match
        if max_val > best_val:
            best_val = max_val
            best_match = (max_loc, temp_w, temp_h)
            best_scale = scale
            
            # Early exit nếu tìm thấy match rất tốt
            if max_val >= early_exit_conf:
                logger.debug(f"⚡ Dừng sớm ở tỉ lệ {scale:.2f} (độ tin cậy={max_val:.4f})")
                break
    
    # Kiểm tra threshold
    if best_val >= threshold and best_match:
        top_left, w, h = best_match
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        
        result = {
            'found': True,
            'confidence': best_val,
            'location': (center_x, center_y),
            'bbox': (top_left[0], top_left[1], w, h),
            'scale': best_scale
        }
        logger.info(f"✅ Tìm thấy ở tỉ lệ={best_scale:.2f}, độ tin cậy={best_val:.4f}, tâm=({center_x}, {center_y})")

        # Debug visualization
        if debug:
            debug_img = screen_bgr.copy()
            cv2.rectangle(debug_img, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 3)
            cv2.circle(debug_img, (center_x, center_y), 8, (0, 0, 255), -1)

            text = f"Conf: {best_val:.3f} | Scale: {best_scale:.2f}"
            cv2.putText(debug_img, text, (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            debug_filename = f"debug_{os.path.basename(template_path).split('.')[0]}.png"
            cv2.imwrite(debug_filename, debug_img)
            logger.info(f"💾 Đã lưu {debug_filename}")
    else:
        logger.debug(f"❌ Không tìm thấy (độ tin cậy tốt nhất={best_val:.4f} < ngưỡng={threshold})")

        # Debug visualization cho failed match
        if debug and best_match:
            debug_img = screen_bgr.copy()
            top_left, w, h = best_match
            cv2.rectangle(debug_img, top_left, (top_left[0] + w, top_left[1] + h), (0, 0, 255), 3)

            text = f"LOW: {best_val:.3f} | Scale: {best_scale:.2f}"
            cv2.putText(debug_img, text, (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            debug_filename = f"debug_{os.path.basename(template_path).split('.')[0]}_failed.png"
            cv2.imwrite(debug_filename, debug_img)
            logger.info(f"💾 Đã lưu {debug_filename}")
    
    return result

# ============================================
# HÀM ADB
# ============================================

def adb_screencap_bytes():
    """Chụp ảnh màn hình qua ADB"""
    p = subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("adb chụp màn hình thất bại")
    return p.stdout

def adb_tap(x, y, randomize=True):
    """Tap với random offset"""
    if randomize:
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)
    
    time.sleep(random.uniform(0.01, 0.03))
    subprocess.run(["adb", "shell", "input", "tap", str(int(x)), str(int(y))])
    logger.info(f"👆 Chạm tại ({int(x)}, {int(y)})")
    _screenshot_buffer.invalidate()

def adb_swipe(x1, y1, x2, y2, duration_ms=200, randomize=True):
    """Swipe với random offset"""
    if randomize:
        x1 += random.randint(-3, 3)
        y1 += random.randint(-3, 3)
        x2 += random.randint(-3, 3)
        y2 += random.randint(-3, 3)
    
    subprocess.run(["adb", "shell", "input", "swipe", 
                    str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms))])
    logger.info(f"👉 Vuốt ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
    _screenshot_buffer.invalidate()

def adb_back():
    """Back button"""
    time.sleep(random.uniform(0.01, 0.03))
    subprocess.run(["adb", "shell", "input", "keyevent", "BACK"])
    logger.info("⬅️  Quay lại")
    _screenshot_buffer.invalidate()

# ============================================
# HÀM CHỤP MÀN HÌNH
# ============================================

def load_screenshot_bgr(use_cache=True, force_refresh=False):
    """Load screenshot với caching"""
    if not use_cache:
        data = adb_screencap_bytes()
        img = Image.open(io.BytesIO(data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    return _screenshot_buffer.get(force_refresh=force_refresh)

# ============================================
# HÀM CẤP CAO - Phiên bản đa tỉ lệ
# ============================================

def click_task_title(screen_bgr=None, max_attempts=2, debug=False, 
                     template_path=r"./templates/item_nv.jpg"):
    """
    Click tiêu đề nhiệm vụ - PHIÊN BẢN ĐA TỈ LỆ
    """
    logger.info("🔍 Tìm tiêu đề nhiệm vụ...")
    
    time.sleep(random.uniform(0.05, 0.15))
    
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    for attempt in range(max_attempts):
        try:
            # Dùng scales tùy chỉnh cho item_nv
            scales = TEMPLATE_SCALES.get('item_nv', DEFAULT_SCALES)
            
            result = match_template_multiscale(
                screen_bgr, template_path, 
                threshold=0.6, 
                scales=scales,
                debug=debug
            )
            
            if result['found']:
                center_x, center_y = result['location']

                # Áp dụng offset đặc biệt cho tiêu đề
                offset_left = 110
                click_x = center_x - offset_left
                click_y = result['bbox'][1] + int(result['bbox'][3] * 0.35)

                logger.info(f"✅ Tiêu đề đã tìm thấy (độ tin cậy={result['confidence']:.3f}, tỉ lệ={result['scale']:.2f})")
                logger.info(f"👆 Nhấn tại ({click_x}, {click_y})")

                if not debug:
                    adb_tap(click_x, click_y, randomize=True)

                return True

            logger.debug(f"Lần thử thứ {attempt+1}/{max_attempts} không thành công")

        except Exception as e:
            logger.error(f"Lỗi ở lần thử thứ {attempt+1}: {e}")
        
        if attempt < max_attempts - 1:
            time.sleep(random.uniform(0.1, 0.2))
            screen_bgr = load_screenshot_bgr(force_refresh=True)
    
    logger.error("❌ Không tìm thấy tiêu đề nhiệm vụ!")
    return False

def click_confirm_button(screen_bgr=None, max_attempts=2, debug=False,
                         template_path=r"./templates/btn_xacnhan.jpg"):
    """
    Click vào nút xác nhận - PHIÊN BẢN ĐA TỈ LỆ
    
    Args:
        screen_bgr: Ảnh màn hình dạng BGR (numpy array), nếu None sẽ chụp ảnh mới
        max_attempts: Số lần thử tối đa (mặc định: 2)
        debug: Chế độ debug - lưu ảnh debug thay vì click (mặc định: False)
        template_path: Đường dẫn đến ảnh template nút xác nhận
        
    Returns:
        True nếu tìm thấy và click thành công
        False nếu không tìm thấy hoặc có lỗi
    """
    logger.info("🔍 Đang tìm nút xác nhận...")
    
    time.sleep(random.uniform(0.05, 0.1))
    
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    for attempt in range(max_attempts):
        try:
            scales = TEMPLATE_SCALES.get('btn_xacnhan', DEFAULT_SCALES)
            
            result = match_template_multiscale(
                screen_bgr, template_path,
                threshold=0.65,
                scales=scales,
                debug=debug
            )
            
            if result['found']:
                click_x, click_y = result['location']
                
                logger.info(f"✅ Nút xác nhận đã tìm thấy (độ tin cậy={result['confidence']:.3f}, tỉ lệ={result['scale']:.2f})")

                if not debug:
                    adb_tap(click_x, click_y, randomize=True)

                return True
            
            logger.debug(f"Lần thử thứ {attempt+1}/{max_attempts} không thành công")
            
        except Exception as e:
            logger.error(f"Lỗi ở lần thử thứ {attempt+1}: {e}")
        
        if attempt < max_attempts - 1:
            time.sleep(random.uniform(0.1, 0.15))
            screen_bgr = load_screenshot_bgr(force_refresh=True)
    
    logger.error("❌ Không tìm thấy nút xác nhận!")
    return False

def check_btn_xn(screen_bgr=None, threshold=0.7, 
                 template_path=r"./templates/btn_xacnhan.jpg", debug=False):
    """
    Kiểm tra nút xác nhận có hiện không - MULTI-SCALE VERSION
    """
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    scales = TEMPLATE_SCALES.get('btn_xacnhan', DEFAULT_SCALES)
    
    result = match_template_multiscale(
        screen_bgr, template_path,
        threshold=threshold,
        scales=scales,
        debug=debug
    )
    
    if result['found']:
        logger.info(f"✅ Nút xác nhận đã tìm thấy! (độ tin cậy={result['confidence']:.3f}, tỉ lệ={result['scale']:.2f})")
        return True
    else:
        return False

def check_captra(screen_bgr=None, threshold=0.5, 
                 template_path=r"./templates/captra.jpg", debug=False):
    """
    Kiểm tra captcha - PHIÊN BẢN ĐA TỈ LỆ
    """
    logger.info(f"🔍 Đang kiểm tra captcha (ngưỡng={threshold})...")
    
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    scales = TEMPLATE_SCALES.get('captra', DEFAULT_SCALES)
    
    result = match_template_multiscale(
        screen_bgr, template_path,
        threshold=threshold,
        scales=scales,
        early_exit_conf=0.9,
        debug=debug
    )
    
    if result['found']:
        logger.info(f"✅ Đã phát hiện captcha! (độ tin cậy={result['confidence']:.3f}, tỉ lệ={result['scale']:.2f})")
        return True
    else:
        logger.info(f"❌ Không tìm thấy captcha (độ tin cậy tốt nhất={result['confidence']:.3f})")
        return False

# ============================================
# HÀM TRỢ GIÚP: Hàm kiểm tra tổng quát
# ============================================

def check_template(template_path, screen_bgr=None, threshold=0.6, 
                   scales=None, debug=False, template_name=None):
    """
    🎯 Hàm tổng quát để kiểm tra bất kỳ template nào
    
    Usage:
        check_template("./templates/button.jpg", threshold=0.7)
    """
    if template_name is None:
        template_name = os.path.basename(template_path)
    
    logger.info(f"🔍 Đang kiểm tra {template_name} (ngưỡng={threshold})...")
    
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    result = match_template_multiscale(
        screen_bgr, template_path,
        threshold=threshold,
        scales=scales or DEFAULT_SCALES,
        debug=debug
    )
    
    if result['found']:
        logger.info(f"✅ {template_name} đã được tìm thấy! (độ tin cậy={result['confidence']:.3f}, tỉ lệ={result['scale']:.2f})")
    else:
        logger.info(f"❌ {template_name} không tìm thấy (độ tin cậy tốt nhất={result['confidence']:.3f})")
    
    return result

# ============================================
# KHỞI TẠO
# ============================================

def preload_templates():
    """Pre-load tất cả templates vào cache"""
    templates = {
        'item_nv': (r"./templates/item_nv.jpg", TEMPLATE_SCALES.get('item_nv')),
        'btn_xacnhan': (r"./templates/btn_xacnhan.jpg", TEMPLATE_SCALES.get('btn_xacnhan')),
        'captra': (r"./templates/captra.jpg", TEMPLATE_SCALES.get('captra')),
    }
    
    logger.info("🔄 Pre-loading templates...")
    for name, (path, scales) in templates.items():
        if os.path.exists(path):
            _template_cache.get(path, scales=scales)
    logger.info("✅ Đã nạp trước tất cả templates!")

# Pre-load khi import module
try:
    preload_templates()
except Exception as e:
    logger.warning(f"Không thể nạp trước templates: {e}")

# ============================================
# MAIN TEST
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 KIỂM TRA TÌM KIẾM TEMPLATE ĐA TỈ LỆ")
    print("=" * 60)
    
    try:
        # Load screenshot
        screen = load_screenshot_bgr()
        print(f"✅ Đã tải ảnh màn hình: {screen.shape}")
        
        # Test 1: Check captcha
        print("\n📋 Test 1: Kiểm tra Captcha")
        print("-" * 60)
        check_captra(screen, threshold=0.5, debug=True)
        
        # Test 2: Check button xác nhận
        print("\n📋 Test 2: Kiểm tra nút Xác Nhận")
        print("-" * 60)
        check_btn_xn(screen, threshold=0.7, debug=True)
        
        # Test 3: Generic check
        print("\n📋 Test 3: Kiểm tra tổng quát")
        print("-" * 60)
        result = check_template(
            "./templates/captra.jpg",
            screen_bgr=screen,
            threshold=0.5,
            debug=True,
            template_name="Nút Captcha"
        )
        print(f"Kết quả: {result}")
        
        print("\n" + "=" * 60)
        print("✅ ĐÃ HOÀN THÀNH TẤT CẢ BÀI KIỂM TRA!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()