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
# ADVANCED OPTIMIZATION: Multi-level Caching
# ============================================

class TemplateCache:
    """Cache template với pre-computed scaled versions"""
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get(self, path, scales=[0.9, 1.0, 1.1]):
        """Lấy template đã scale sẵn từ cache"""
        cache_key = (path, tuple(scales))
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Load và pre-compute tất cả scales
            template = cv2.imread(path)
            if template is None:
                return None
            
            scaled_templates = []
            for scale in scales:
                if scale == 1.0:
                    scaled_templates.append((template, scale))
                else:
                    w = int(template.shape[1] * scale)
                    h = int(template.shape[0] * scale)
                    resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)
                    scaled_templates.append((resized, scale))
            
            self._cache[cache_key] = scaled_templates
            logger.info(f"✅ Cached template: {path} với {len(scales)} scales")
            return scaled_templates
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()

class ScreenshotBuffer:
    """Buffer để tái sử dụng screenshot trong cùng 1 cycle"""
    def __init__(self, ttl=0.5):
        self._buffer = None
        self._timestamp = 0
        self._ttl = ttl  # Time to live (giây)
        self._lock = threading.Lock()
    
    def get(self, force_refresh=False):
        """Lấy screenshot, tái sử dụng nếu còn fresh"""
        with self._lock:
            current_time = time.time()
            
            # Kiểm tra cache còn valid không
            if not force_refresh and self._buffer is not None:
                age = current_time - self._timestamp
                if age < self._ttl:
                    logger.debug(f"♻️  Reusing screenshot (age: {age:.2f}s)")
                    return self._buffer
            
            # Capture mới
            logger.debug("📸 Capturing new screenshot")
            data = adb_screencap_bytes()
            img = Image.open(io.BytesIO(data))
            self._buffer = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self._timestamp = current_time
            
            return self._buffer
    
    def invalidate(self):
        """Vô hiệu hóa cache (sau khi tap/swipe)"""
        with self._lock:
            self._timestamp = 0

# Global instances
_template_cache = TemplateCache()
_screenshot_buffer = ScreenshotBuffer(ttl=0.3)  # 300ms TTL
_executor = ThreadPoolExecutor(max_workers=2)

# Pre-load templates at startup
def preload_templates():
    """Pre-load tất cả templates vào cache khi khởi động"""
    templates = [
        r"./templates/item_nv.jpg",
        r"./templates/btn_xacnhan.jpg"
    ]
    
    logger.info("🔄 Pre-loading templates...")
    for path in templates:
        if os.path.exists(path):
            _template_cache.get(path)
    logger.info("✅ Templates pre-loaded!")

# ============================================
# ANTI-DETECTION: Optimized Functions
# ============================================

def random_sleep(min_sec=0.2, max_sec=0.5):
    """Ngủ ngẫu nhiên - ULTRA FAST"""
    time.sleep(random.uniform(min_sec, max_sec))

def add_random_offset(x, y, max_offset=3):
    """Thêm độ lệch ngẫu nhiên"""
    return x + random.randint(-max_offset, max_offset), y + random.randint(-max_offset, max_offset)

def simulate_human_delay():
    """Delay cực ngắn 10-30ms"""
    time.sleep(random.uniform(0.01, 0.03))

# ============================================
# ADB Functions - Optimized
# ============================================

def adb_screencap_bytes():
    """Chụp ảnh màn hình"""
    p = subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("adb screencap failed")
    return p.stdout

def adb_tap(x, y, randomize=True):
    """Tap với invalidate buffer"""
    if randomize:
        x, y = add_random_offset(x, y, max_offset=5)
    
    simulate_human_delay()
    subprocess.run(["adb", "shell", "input", "tap", str(int(x)), str(int(y))])
    logger.info(f"Tap ({int(x)}, {int(y)})")
    
    # Invalidate screenshot buffer sau khi tap
    _screenshot_buffer.invalidate()

def adb_swipe(x1, y1, x2, y2, duration_ms=200, randomize=True):
    """Swipe với invalidate buffer"""
    if randomize:
        x1, y1 = add_random_offset(x1, y1)
        x2, y2 = add_random_offset(x2, y2)
    
    subprocess.run(["adb", "shell", "input", "swipe", 
                    str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms))])
    _screenshot_buffer.invalidate()

def adb_back():
    """Back button"""
    simulate_human_delay()
    subprocess.run(["adb", "shell", "input", "keyevent", "BACK"])
    _screenshot_buffer.invalidate()

# ============================================
# Screenshot Functions - Cached
# ============================================

def load_screenshot_bgr(use_cache=True, force_refresh=False):
    """
    Load screenshot với caching thông minh
    - use_cache=True: Dùng buffer nếu còn fresh
    - force_refresh=True: Bắt buộc capture mới
    """
    if not use_cache:
        data = adb_screencap_bytes()
        img = Image.open(io.BytesIO(data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    return _screenshot_buffer.get(force_refresh=force_refresh)

# ============================================
# Template Matching - Ultra Optimized
# ============================================

def match_template_fast(screen_bgr, template_path, threshold=0.6):
    """
    Template matching CỰC NHANH với pre-computed scales
    Trả về (best_match, best_val) hoặc (None, 0)
    """
    scaled_templates = _template_cache.get(template_path)
    if scaled_templates is None:
        return None, 0
    
    best_match = None
    best_val = 0
    
    # Match với tất cả pre-computed scales
    for template, scale in scaled_templates:
        result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            h, w = template.shape[:2]
            best_match = (max_loc, w, h, scale)
        
        # Early exit nếu đã tìm thấy match rất tốt
        if max_val > 0.85:
            break
    
    if best_val < threshold:
        return None, best_val
    
    return best_match, best_val

# ============================================
# High-level Functions - Optimized
# ============================================

def click_task_title(screen_bgr=None, max_attempts=2, debug=False, 
                     template_path=r"./templates/item_nv.jpg"):
    """
    Click tiêu đề nhiệm vụ - ULTRA OPTIMIZED
    - Nhận screen_bgr từ bên ngoài để tránh capture lại
    - Sử dụng pre-computed templates
    """
    logger.info("🔍 Tìm tiêu đề...")
    
    time.sleep(random.uniform(0.05, 0.15))
    
    # Dùng screen đã có hoặc load mới
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    for attempt in range(max_attempts):
        try:
            best_match, best_val = match_template_fast(screen_bgr, template_path)
            
            if best_match:
                top_left, w, h, scale = best_match
                
                offset_left = 110
                click_x = top_left[0] + w // 2 - offset_left
                click_y = top_left[1] + int(h * 0.35)
                
                logger.info(f"✅ Found (conf={best_val:.2f}) -> ({click_x}, {click_y})")
                
                if debug:
                    debug_img = screen_bgr.copy()
                    cv2.rectangle(debug_img, top_left, (top_left[0] + w, top_left[1] + h), (255,0,255), 2)
                    cv2.circle(debug_img, (int(click_x), int(click_y)), 8, (0,0,255), -1)
                    cv2.imwrite('debug_click_task.png', debug_img)
                    return True
                
                adb_tap(click_x, click_y, randomize=True)
                return True
            
            logger.debug(f"Not found (attempt {attempt+1}, conf={best_val:.2f})")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
        
        if attempt < max_attempts - 1:
            time.sleep(random.uniform(0.1, 0.2))
            # Refresh screenshot cho attempt tiếp theo
            screen_bgr = load_screenshot_bgr(force_refresh=True)
    
    return False

def click_confirm_button(screen_bgr=None, max_attempts=2, debug=False,
                         template_path=r"./templates/btn_xacnhan.jpg"):
    """
    Click nút xác nhận - ULTRA OPTIMIZED
    """
    logger.info("🔍 Tìm nút xác nhận...")
    
    time.sleep(random.uniform(0.05, 0.1))
    
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    for attempt in range(max_attempts):
        try:
            best_match, best_val = match_template_fast(screen_bgr, template_path)
            
            if best_match:
                top_left, w, h, scale = best_match
                click_x = top_left[0] + w // 2
                click_y = top_left[1] + h // 2
                
                logger.info(f"✅ Found (conf={best_val:.2f}) -> ({click_x}, {click_y})")
                
                if debug:
                    debug_img = screen_bgr.copy()
                    cv2.rectangle(debug_img, top_left, (top_left[0] + w, top_left[1] + h), (255,0,255), 2)
                    cv2.circle(debug_img, (int(click_x), int(click_y)), 8, (0,0,255), -1)
                    cv2.imwrite('debug_click_confirm.png', debug_img)
                    return True
                
                adb_tap(click_x, click_y, randomize=True)
                return True
            
            logger.debug(f"Not found (attempt {attempt+1}, conf={best_val:.2f})")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
        
        if attempt < max_attempts - 1:
            time.sleep(random.uniform(0.1, 0.15))
            screen_bgr = load_screenshot_bgr(force_refresh=True)

    logger.error("❌ Không tìm thấy nút xác nhận!")
    return False

def check_btn_xn(screen_bgr=None, threshold=0.7, 
                 template_path=r"./templates/btn_xacnhan.jpg", debug=False):
    """
    Kiểm tra button - ULTRA FAST
    """
    if screen_bgr is None:
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    try:
        best_match, best_val = match_template_fast(screen_bgr, template_path, threshold)
        
        if best_match:
            logger.info(f"✅ Button found (conf={best_val:.2f})")
            
            if debug:
                debug_img = screen_bgr.copy()
                top_left, w, h, scale = best_match
                cv2.rectangle(debug_img, top_left, (top_left[0] + w, top_left[1] + h), (0,255,0), 2)
                cv2.imwrite('debug_check_btn_xn.png', debug_img)
            
            return True
        else:
            logger.debug(f"Button not found (conf={best_val:.2f})")
            return False
            
    except Exception as e:
        logger.error(f"Error checking button: {e}")
        return False

def check_captra(screen_bgr=None, threshold=0.5, 
                 template_path=r"./templates/captra.jpg", debug=False):
    """
    Kiểm tra button captra - Tự động thử nhiều scale để phù hợp với zoom level
    """
    logger.info(f"🔍 check_captra() called with threshold={threshold}")
    
    if screen_bgr is None:
        logger.info("📸 Loading screenshot...")
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    try:
        # Kiểm tra kích thước screen
        screen_h, screen_w = screen_bgr.shape[:2]
        logger.info(f"📺 Screen size: {screen_w}x{screen_h}")
        
        logger.info(f"📂 Loading template: {template_path}")
        template_original = cv2.imread(template_path)
        
        if template_original is None:
            logger.error(f"❌ Không đọc được template: {template_path}")
            return False
        
        temp_h, temp_w = template_original.shape[:2]
        logger.info(f"🖼️ Template size: {temp_w}x{temp_h}")
        
        # ĐÂY LÀ PHẦN QUAN TRỌNG: Thử nhiều scale từ 50% đến 150%
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        best_val = 0
        best_match = None
        best_scale = 1.0
        
        logger.info(f"🔄 Đang thử {len(scales)} scales khác nhau...")
        
        for scale in scales:
            # Resize template
            new_w = int(temp_w * scale)
            new_h = int(temp_h * scale)
            
            # Bỏ qua nếu template lớn hơn màn hình
            if new_w > screen_w or new_h > screen_h:
                logger.debug(f"⏭️ Skip scale {scale:.2f} (quá lớn: {new_w}x{new_h})")
                continue
            
            # Resize template
            template = cv2.resize(template_original, (new_w, new_h), 
                                 interpolation=cv2.INTER_LINEAR)
            
            # Match template
            result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            logger.debug(f"📏 Scale {scale:.2f} ({new_w}x{new_h}) -> conf={max_val:.4f}")
            
            # Cập nhật best match
            if max_val > best_val:
                best_val = max_val
                best_match = (max_loc, new_w, new_h)
                best_scale = scale
                
                # Early exit nếu tìm thấy match rất tốt
                if max_val > 0.9:
                    logger.info(f"⚡ Tìm thấy match tốt sớm tại scale {scale:.2f}")
                    break
        
        logger.info(f"📊 Best match: conf={best_val:.4f} at scale={best_scale:.2f} (threshold: {threshold})")
        
        # Kiểm tra threshold
        if best_val >= threshold:
            logger.info(f"✅ Button found! (confidence={best_val:.4f}, scale={best_scale:.2f})")
            
            if debug and best_match:
                debug_img = screen_bgr.copy()
                top_left, w, h = best_match
                
                # Vẽ khung
                cv2.rectangle(debug_img, top_left, 
                            (top_left[0] + w, top_left[1] + h), (0, 255, 0), 3)
                
                # Vẽ text
                text = f"Conf: {best_val:.2f} | Scale: {best_scale:.2f}"
                cv2.putText(debug_img, text, 
                           (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imwrite('debug_check_captra.png', debug_img)
                logger.info("💾 Đã lưu debug_check_captra.png")
            
            return True
        else:
            logger.warning(f"❌ Button not found (best conf={best_val:.4f} < {threshold})")
            
            if debug and best_match:
                # Vẫn lưu ảnh debug để xem
                debug_img = screen_bgr.copy()
                top_left, w, h = best_match
                
                cv2.rectangle(debug_img, top_left, 
                            (top_left[0] + w, top_left[1] + h), (0, 0, 255), 3)
                
                text = f"LOW: {best_val:.2f} | Scale: {best_scale:.2f}"
                cv2.putText(debug_img, text, 
                           (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imwrite('debug_check_captra.png', debug_img)
                logger.info("💾 Đã lưu debug_check_captra.png (not found)")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking button: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# Nếu muốn tích hợp vào TemplateCache để tăng tốc
def check_captra_cached(screen_bgr=None, threshold=0.5, 
                        template_path=r"./templates/captra.jpg", debug=False):
    """
    Version dùng cache - nhanh hơn nếu gọi nhiều lần
    Sử dụng TemplateCache với nhiều scales tùy chỉnh
    """
    logger.info(f"🔍 check_captra_cached() called with threshold={threshold}")
    
    if screen_bgr is None:
        logger.info("📸 Loading screenshot...")
        screen_bgr = load_screenshot_bgr(use_cache=True)
    
    try:
        # Kiểm tra kích thước screen
        screen_h, screen_w = screen_bgr.shape[:2]
        logger.info(f"📺 Screen size: {screen_w}x{screen_h}")
        
        # Định nghĩa scales phù hợp
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        # Lấy scaled templates từ cache
        scaled_templates = _template_cache.get(template_path, scales=scales)
        
        if scaled_templates is None:
            logger.error(f"❌ Không đọc được template: {template_path}")
            return False
        
        best_val = 0
        best_match = None
        best_scale = 1.0
        
        logger.info(f"🔄 Đang thử {len(scales)} scales khác nhau...")
        
        # Match với tất cả pre-computed scales
        for template, scale in scaled_templates:
            h, w = template.shape[:2]
            
            # Bỏ qua nếu template lớn hơn màn hình
            if w > screen_w or h > screen_h:
                logger.debug(f"⏭️ Skip scale {scale:.2f} (quá lớn: {w}x{h})")
                continue
            
            # Match template
            result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            logger.debug(f"🔍 Scale {scale:.2f} ({w}x{h}) -> conf={max_val:.4f}")
            
            # Cập nhật best match
            if max_val > best_val:
                best_val = max_val
                best_match = (max_loc, w, h)
                best_scale = scale
                
                # Early exit nếu tìm thấy match rất tốt
                if max_val > 0.9:
                    logger.info(f"⚡ Tìm thấy match tốt sớm tại scale {scale:.2f}")
                    break
        
        logger.info(f"📊 Best match: conf={best_val:.4f} at scale={best_scale:.2f} (threshold: {threshold})")
        
        # Kiểm tra threshold
        if best_val >= threshold:
            logger.info(f"✅ Button found! (confidence={best_val:.4f}, scale={best_scale:.2f})")
            
            if debug and best_match:
                debug_img = screen_bgr.copy()
                top_left, w, h = best_match
                
                # Vẽ khung
                cv2.rectangle(debug_img, top_left, 
                            (top_left[0] + w, top_left[1] + h), (0, 255, 0), 3)
                
                # Vẽ text
                text = f"Conf: {best_val:.2f} | Scale: {best_scale:.2f}"
                cv2.putText(debug_img, text, 
                           (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imwrite('debug_check_captra_cached.png', debug_img)
                logger.info("💾 Đã lưu debug_check_captra_cached.png")
            
            return True
        else:
            logger.warning(f"❌ Button not found (best conf={best_val:.4f} < {threshold})")
            
            if debug and best_match:
                # Vẫn lưu ảnh debug để xem
                debug_img = screen_bgr.copy()
                top_left, w, h = best_match
                
                cv2.rectangle(debug_img, top_left, 
                            (top_left[0] + w, top_left[1] + h), (0, 0, 255), 3)
                
                text = f"LOW: {best_val:.2f} | Scale: {best_scale:.2f}"
                cv2.putText(debug_img, text, 
                           (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imwrite('debug_check_captra_cached.png', debug_img)
                logger.info("💾 Đã lưu debug_check_captra_cached.png (not found)")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking button: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
# ============================================
# Legacy Functions (kept for compatibility)
# ============================================

def find_text_ocr(screen_bgr, text_list, conf_thresh=40, lang="rus"):
    """OCR - kept for compatibility"""
    rgb = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    
    config = "--psm 6"
    if lang:
        config += f" -l {lang}"
    
    try:
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config=config)
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return None
    
    n = len(data['text'])
    for i in range(n):
        txt = str(data['text'][i]).strip()
        conf = -1
        try:
            conf = int(float(data['conf'][i]))
        except:
            pass
        
        if conf >= conf_thresh:
            for search_text in text_list:
                if search_text.lower() in txt.lower():
                    x = int(data['left'][i] + data['width'][i]/2)
                    y = int(data['top'][i] + data['height'][i]/2)
                    return {'center': (x, y), 'conf': conf, 'text': txt, 'matched': search_text}
    
    return None

# ============================================
# Initialization
# ============================================

# Pre-load templates khi import module
try:
    preload_templates()
except Exception as e:
    logger.warning(f"Could not preload templates: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 STARTING TEST")
    print("=" * 50)
    
    try:
        screen = load_screenshot_bgr()
        print(f"✅ Screenshot loaded: {screen.shape}")
        
        print("\n🔍 Running check_captra...")
        if check_captra_cached(screen, debug=True):
            print("✅ Test OK!")
        else:
            print("❌ Test FAILED!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)
    print("🏁 TEST FINISHED")
    print("=" * 50)