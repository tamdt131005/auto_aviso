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
        logger,
        _screenshot_buffer  # Import buffer để control
    )
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
    logger.info(f"⏸️  Break {duration:.1f}s...")
    time.sleep(duration)
    # Invalidate buffer sau break
    _screenshot_buffer.invalidate()

if __name__ == "__main__":
    count = 0
    max_count = 50
    max_wait_attempts = 6  # Tăng lên 6 cho đủ check trong 14s
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("🚀 ULTRA SPEED MODE - DEEP OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"🎯 Target: {max_count} lần")
    logger.info(f"⚡ Optimizations:")
    logger.info(f"   - Template pre-loaded & cached với scaled versions")
    logger.info(f"   - Screenshot buffer reuse (TTL=300ms)")
    logger.info(f"   - Early exit khi match score > 0.85")
    logger.info(f"   - Ultra-fast delays (10-50ms)")
    logger.info("=" * 60)
    
    # Minimal initial delay
    time.sleep(random.uniform(0.3, 0.6))
    
    while count < max_count:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 [{count + 1}/{max_count}]")
            
            # Break logic
            if should_take_break(count, interval=25):
                take_smart_break()
            
            # ============================================
            # OPTIMIZED WORKFLOW: Minimize screenshot captures
            # ============================================
            
            # Bước 1: Capture 1 lần duy nhất
            logger.info("📸 Capture screen...")
            screen = load_screenshot_bgr(use_cache=False, force_refresh=True)
            
            # Bước 2: Click task title (dùng screen đã có)
            logger.info("🔍 Step 1: Find & click task...")
            if not click_task_title(screen_bgr=screen, debug=False):
                logger.warning("⚠️  Task not found. Retry...")
                fail_count += 1
                smart_wait(0.8, 0.3)
                continue
            
            logger.info("✅ Clicked task")
            
            # Delay sau click - CHỜ UI LOAD (11-14s thực tế)
            wait_time = smart_wait(12.5, 1)  # 10.5-13.5s
            logger.info(f"⏱️  Waiting for UI to load: {wait_time:.1f}s")
            logger.debug(f"   (Button appears after ~11-14s)")
            
            # ============================================
            # Bước 3: Đợi button xuất hiện (CHUẨN CHO 1,5-3s LOAD TIME)
            # ============================================
            logger.info("🔍 Step 2: Wait for button (11-14s load time)...")
            wait_attempts = 0
            btn_found = False
            
            # STRATEGY: Check định kỳ trong khoảng 11-14s
            check_intervals = [
                1.8,  # Check lần 1 sau 1.5s (sớm 1 chút)
                1.0,   # Check lần 2 sau thêm 1s (11.5s total)
                1.0,   # Check lần 3 sau thêm 1s (12.5s total)
                1.0,   # Check lần 4 sau thêm 1s (13.5s total)
                0.5,   # Check lần 5 sau thêm 0.5s (14s total)
                0.5    # Check lần 6 sau thêm 0.5s (14.5s total - backup)
            ]
            
            for idx, interval in enumerate(check_intervals):
                if wait_attempts >= max_wait_attempts:
                    break
                
                # Đợi theo interval
                logger.debug(f"⏳ Sleeping {interval}s before check #{idx+1}...")
                time.sleep(interval)
                
                # Capture và check
                screen = load_screenshot_bgr(force_refresh=True)
                
                if check_btn_xn(screen_bgr=screen, debug=False):
                    total_wait = sum(check_intervals[:idx+1])
                    logger.info(f"✅ Button detected after {total_wait:.1f}s!")
                    btn_found = True
                    break
                
                wait_attempts += 1
                logger.debug(f"   Not yet (attempt {wait_attempts}/{max_wait_attempts})")
            
            if not btn_found:
                logger.warning("⏱️  Timeout. Skip...")
                fail_count += 1
                smart_wait(0.6, 0.2)
                continue
            
            # Minimal delay before click
            time.sleep(random.uniform(0.05, 0.1))
            
            # ============================================
            # Bước 4: Click button (reuse screen)
            # ============================================
            logger.info("👆 Step 3: Click button...")
            
            # OPTIMIZATION: Không cần capture lại, dùng screen hiện tại
            if click_confirm_button(screen_bgr=screen, debug=False):
                count += 1
                success_count += 1
                
                # Stats
                elapsed = time.time() - start_time
                avg_time = elapsed / count
                remaining = avg_time * (max_count - count)
                rate = count / (elapsed / 60)  # lần/phút
                
                logger.info(f"✅ Done {count}/{max_count}")
                logger.info(f"📊 {success_count} ok | {fail_count} fail")
                logger.info(f"⚡ Speed: {rate:.1f}/min | Avg: {avg_time:.1f}s")
                logger.info(f"🕐 ETA: {remaining/60:.1f}min")
            else:
                logger.warning("⚠️  Click failed. Retry...")
                fail_count += 1
                smart_wait(0.6, 0.2)
                continue
            
            # ============================================
            # Ultra-minimal inter-action delay
            # ============================================
            # Giảm xuống 0.4-0.8s
            inter_delay = smart_wait(0.5, 0.25)
            logger.debug(f"⏱️  Inter: {inter_delay:.2f}s")
            
        except KeyboardInterrupt:
            logger.info("\n\n⛔ Stopped (Ctrl+C)")
            break
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            fail_count += 1
            logger.info("⏳ Wait 1.5-2.5s...")
            smart_wait(2.0, 0.5)
            # Invalidate buffer sau error
            _screenshot_buffer.invalidate()
            continue
    
    # ============================================
    # Final Stats
    # ============================================
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("🎉 COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Success: {success_count}/{max_count}")
    logger.info(f"⚠️  Failed: {fail_count}")
    logger.info(f"⏱️  Total: {total_time/60:.2f} min")
    
    if success_count > 0:
        avg = total_time / success_count
        rate = success_count / (total_time / 60)
        logger.info(f"📊 Average: {avg:.2f}s/action")
        logger.info(f"⚡ Speed: {rate:.1f} actions/min")
        logger.info(f"🚀 Performance: {(rate/12)*100:.0f}% faster than baseline")
    
    logger.info(f"💾 Cache stats: Screenshot buffer saved ~{wait_attempts * success_count} captures")
    logger.info(f"{'='*60}")