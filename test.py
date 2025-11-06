import sys
import os
import time
import random

try:
    from models import (
        load_screenshot_bgr,
        click_task_title,
        click_confirm_button,
        check_btn_xn,
        random_sleep,
        check_captra,
        check_captra_cached,
        logger,
        _screenshot_buffer
    )
    from amthanh import start_alert, stop_alert
except ImportError as e:
    print(f"Lỗi import: {e}")
    sys.exit(1) 

def smart_wait(base=0.3, variance=0.15):
    """Ultra fast wait"""
    wait_time = max(0.1, base + random.uniform(-variance, variance))
    time.sleep(wait_time)
    return wait_time

def should_take_break(count, interval=25):
    """Nghỉ sau mỗi 25 lần"""
    return count > 0 and count % interval == 0

def take_smart_break():
    """Nghỉ 2-5 giây"""
    duration = random.uniform(2, 5)
    logger.info(f"⸏️  Break {duration:.1f}s...")
    time.sleep(duration)
    _screenshot_buffer.invalidate()

def wait_and_solve_captcha(max_wait=60, check_interval=2):
    """
    Đợi và giải captcha nếu xuất hiện
    Returns: True nếu captcha đã được giải hoặc không có captcha, False nếu timeout
    """
    logger.info("🔍 Checking for captcha...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        screen = load_screenshot_bgr(force_refresh=True)
        
        if check_captra_cached(screen, threshold=0.5):
            logger.warning("🔒 CAPTCHA DETECTED!")
            start_alert()
            time.sleep(3)
            stop_alert()
            
            # Đợi user giải captcha
            logger.info("⏳ Please solve the captcha...")
            logger.info("⏳ Waiting up to 60s for captcha to be solved...")
            
            captcha_start = time.time()
            while time.time() - captcha_start < 60:
                time.sleep(2)
                screen = load_screenshot_bgr(force_refresh=True)
                
                # Kiểm tra captcha đã biến mất chưa
                if not check_captra_cached(screen, threshold=0.5):
                    logger.info("✅ Captcha solved!")
                    _screenshot_buffer.invalidate()
                    return True
                
                logger.debug("⏳ Still waiting for captcha solution...")
            
            logger.error("❌ Captcha timeout (60s)")
            return False
        
        # Không có captcha, OK
        logger.debug("✅ No captcha detected")
        return True
    
    return True

if __name__ == "__main__":
    count = 0
    max_count = 50
    max_wait_attempts = 7  # Tăng lên 7 lần check (tổng ~16s)
    
    success_count = 0
    fail_count = 0
    captcha_count = 0
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("🚀 ULTRA SPEED MODE - WITH CAPTCHA HANDLING")
    logger.info("=" * 60)
    logger.info(f"🎯 Target: {max_count} lần")
    logger.info(f"⚡ Optimizations:")
    logger.info(f"   - Template pre-loaded & cached với scaled versions")
    logger.info(f"   - Screenshot buffer reuse (TTL=300ms)")
    logger.info(f"   - Early exit khi match score > 0.85")
    logger.info(f"   - Ultra-fast delays (10-50ms)")
    logger.info(f"   - Smart captcha detection after task click")
    logger.info("=" * 60)
    logger.info("📋 Workflow:")
    logger.info("   Click task → Check captcha → Wait button → Click confirm")
    logger.info("=" * 60)
    
    time.sleep(random.uniform(0.3, 0.6))
    
    while count < max_count:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 [{count + 1}/{max_count}]")
            
            # Break logic
            if should_take_break(count, interval=25):
                take_smart_break()
            
            # ============================================
            # Bước 1: Capture và click task
            # ============================================
            logger.info("📸 Step 1: Capture screen and click task...")
            screen = load_screenshot_bgr(use_cache=False, force_refresh=True)
            
            if not click_task_title(screen_bgr=screen, debug=False):
                logger.warning("⚠️  Task not found. Retry...")
                fail_count += 1
                smart_wait(0.8, 0.3)
                continue
            logger.info("✅ Clicked task")
            
            # ============================================
            # Bước 2: Đợi một chút rồi check captcha
            # ============================================
            logger.info("⏱️  Step 2: Waiting for page load & checking captcha...")
            time.sleep(random.uniform(3.5, 4.5))  # Đợi UI load một chút
            
            # Check và giải captcha nếu có
            if not wait_and_solve_captcha(max_wait=60, check_interval=2):
                logger.error("❌ Failed to solve captcha. Skipping...")
                fail_count += 1
                captcha_count += 1
                smart_wait(1.0, 0.5)
                continue
            
            # Nếu có captcha và đã giải xong, đợi thêm chút
            if captcha_count > 0:
                logger.info("⏳ Waiting for UI to refresh after captcha...")
                time.sleep(random.uniform(1.0, 2.0))
            
            # ============================================
            # Bước 3: Đợi button confirm xuất hiện
            # ============================================
            logger.info("🔍 Step 3: Waiting for confirm button...")
            
            # STRATEGY: Sau khi task load (hoặc captcha solved), button xuất hiện sau 7-14s
            # Tổng thời gian đã đợi: ~4s (load) + ~2s (nếu có captcha)
            # Còn cần đợi: ~7-12s nữa
            
            check_intervals = [
                1.0,   # Check sau 3s (~7s total)
                1.0,   # Check sau 2s (~9s total)
                1.0,   # Check sau 2s (~11s total)
                1.0,   # Check sau 1s (~12s total)
                1.0,   # Check sau 1s (~13s total)
                1.0,   # Check sau 1s (~14s total)
                1.0,   # Check sau 1s (~15s total - backup)
            ]
            
            btn_found = False
            total_waited = 0
            
            for idx, interval in enumerate(check_intervals):
                # Sleep
                time.sleep(interval)
                total_waited += interval
                
                # Capture fresh screen
                screen = load_screenshot_bgr(force_refresh=True)
                
                # Check button
                if check_btn_xn(screen_bgr=screen, debug=False):
                    logger.info(f"✅ Button found after {total_waited:.1f}s!")
                    btn_found = True
                    break
                
                logger.debug(f"⏳ Not yet... ({total_waited:.1f}s waited, attempt {idx+1}/{len(check_intervals)})")
            
            # Nếu không tìm thấy button sau timeout
            if not btn_found:
                logger.warning(f"⏱️  Button timeout after {total_waited:.1f}s. Skip...")
                fail_count += 1
                smart_wait(0.6, 0.2)
                continue
            
            # Minimal delay before click
            time.sleep(random.uniform(0.05, 0.15))
            
            # ============================================
            # Bước 4: Click confirm button
            # ============================================
            logger.info("👆 Step 4: Click confirm button...")
            
            if click_confirm_button(screen_bgr=screen, debug=False):
                count += 1
                success_count += 1
                # Thống kê
                elapsed = time.time() - start_time
                avg_time = elapsed / count
                remaining = avg_time * (max_count - count)
                rate = count / (elapsed / 60)  # lần/phút
                logger.info(f"✅ Hoàn thành {count}/{max_count}")
                logger.info(f"📊 Thành công: {success_count} | Thất bại: {fail_count} | Captcha: {captcha_count}")
                logger.info(f"⚡ Tốc độ: {rate:.1f}/phút | TB: {avg_time:.1f}s/lần")
                logger.info(f"🕐 Ước tính còn lại: {remaining/60:.1f} phút")
            else:
                logger.warning("⚠️  Failed to click button. Retry...")
                fail_count += 1
                smart_wait(0.6, 0.2)
                continue
            
            # ============================================
            # Ultra-minimal inter-action delay
            # ============================================
            inter_delay = smart_wait(0.5, 0.25)
            logger.debug(f"⏱️  Inter-action delay: {inter_delay:.2f}s")
            
        except KeyboardInterrupt:
            logger.info("\n\n⛔ Stopped by user (Ctrl+C)")
            break
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            fail_count += 1
            logger.info("⏳ Waiting 2s before retry...")
            smart_wait(2.0, 0.5)
            _screenshot_buffer.invalidate()
            continue
    
    # ============================================
    # Final Stats
    # ============================================
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("🎉 AUTOMATION COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Success: {success_count}/{max_count}")
    logger.info(f"❌ Failed: {fail_count}")
    logger.info(f"🔒 Captcha encounters: {captcha_count}")
    logger.info(f"⏱️  Total time: {total_time/60:.2f} minutes")
    
    if success_count > 0:
        avg = total_time / success_count
        rate = success_count / (total_time / 60)
        efficiency = (success_count / (success_count + fail_count)) * 100
        
        logger.info(f"📊 Average time: {avg:.2f}s per action")
        logger.info(f"⚡ Speed: {rate:.1f} actions/min")
        logger.info(f"🎯 Success rate: {efficiency:.1f}%")
        
        if captcha_count > 0:
            logger.info(f"🔒 Captcha rate: {(captcha_count/count)*100:.1f}%")
    
    logger.info(f"{'='*60}")