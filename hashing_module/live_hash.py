import cv2
import sqlite3
import hashlib
import time
import os
import numpy as np
from datetime import datetime
from PIL import Image
import imagehash
from hashing_module.config.config import DB_PATH, FRAME_SIZE, TO_GRAY, HASH_ALGO
from hashing_module.db.schema import init_db

# Path to the handshake flag created by HTM
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READY_FLAG_PATH = os.path.join(PROJECT_ROOT, "htm_ready.flag")
def normalize_frame(frame):
    if TO_GRAY:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return frame

def crypto_hash(frame):
    frame_bytes = frame.tobytes()
    if HASH_ALGO == "sha256":
        return hashlib.sha256(frame_bytes).hexdigest()
    elif HASH_ALGO == "blake2b":
        return hashlib.blake2b(frame_bytes, digest_size=32).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm")

def perceptual_hash(frame):
    pil_img = Image.fromarray(frame)
    return str(imagehash.phash(pil_img))

def wait_for_htm_start():
    """Blocks execution until the HTM engine signals it is ready."""
    print(f"[Hashing] Looking for flag at: {READY_FLAG_PATH}", flush=True)
    
    waiting = True
    while waiting:
        if os.path.exists(READY_FLAG_PATH):
            print(f"[Hashing] ✅ Found {READY_FLAG_PATH}! Syncing...", flush=True)
            waiting = False
        else:
            print(f"[Hashing] ⏳ Waiting for HTM start signal...", end="\r", flush=True)
            time.sleep(0.5)

def live_hash(stream_url="rtsp://localhost:8554/camera",
              camera_id="CAM10", video_id="VID010",
              max_frames=None, max_seconds=None):
    
    # 1. WAIT FOR HTM BEFORE CONNECTING
    # This prevents the TCP buffer from filling up while HTM loads models.
    wait_for_htm_start()

    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    # [RECOMMENDED] Enable WAL to prevent locking latency
    conn.execute("PRAGMA journal_mode=WAL;")
    
    cursor = conn.cursor()

    # 2. CONNECT NOW (Fresh Start)
    cap = None
    retry_count = 0
    while retry_count < 10:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"[Hashing] ✅ Connected to {stream_url}")
            break
        
        print(f"[Hashing] ⚠️ Connection failed (Attempt {retry_count+1}/10). Retrying in 1s...")
        time.sleep(1.0)
        retry_count += 1
        
    if cap is None or not cap.isOpened():
        print(f"[Hashing] ❌ FATAL: Could not connect to {stream_url} after retries.")
        conn.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30.0
    print(f"[Hashing] Stream FPS: {fps}")

    frame_idx = 0
    print(f"[Hashing] Starting live hashing on {stream_url}")
    start_time = time.time()
    
    # Timer for the 'read_fresh' logic
    prev_loop_duration = 0.0

    try:
        while True:
            loop_start = time.time()

            # 3. APPLY 'READ_FRESH' LOGIC (Exact copy of HTM logic)
            # If the previous loop took too long (e.g. DB write), skip frames to catch up.
            frames_to_skip = int(prev_loop_duration * fps)
            
            # Cap skip to avoid crazy behavior if glitches happen
            frames_to_skip = max(0, min(frames_to_skip, 100))

            if frames_to_skip > 0:
                # print(f"[Hashing] Skipping {frames_to_skip} frames to sync...") # Uncomment to debug
                for _ in range(frames_to_skip):
                    cap.grab()

            ret, frame = cap.read()
            if not ret:
                print("[Hashing] Stream ended or cannot grab frame. Stopping.")
                break

            # Frame index is incremented by 1 (processed) + skipped frames
            frame_idx += 1 + frames_to_skip
            
            ts_iso = datetime.utcnow().isoformat()

            norm = normalize_frame(frame)
            crypto_h = crypto_hash(norm)
            phash_h = perceptual_hash(norm)

            cursor.execute("""
                INSERT INTO frame_hashes (camera_id, video_id, frame_number, timestamp,
                                          hash, phash, hash_algo,
                                          gray, width, height, serialize_mode, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (camera_id, video_id, frame_idx, ts_iso,
                  crypto_h, phash_h, HASH_ALGO,
                  int(TO_GRAY), FRAME_SIZE[0], FRAME_SIZE[1], "raw", "live"))
            
            # Commit every frame (as per your current logic)
            # Note: WAL mode (PRAGMA above) makes this much faster/safer
            conn.commit()

            if max_frames and frame_idx >= max_frames: break
            if max_seconds and (time.time() - start_time) >= max_seconds: break
            
            # Calculate how long this loop took to inform the NEXT skip
            prev_loop_duration = time.time() - loop_start
                
    except Exception as e:
        print(f"[Hashing] Error: {e}")
    finally:
        cap.release()
        conn.close()
        print(f"[Hashing] Stopped after {frame_idx} frames.")

if __name__ == "__main__":
    live_hash("rtsp://localhost:8554/camera", camera_id="TEST", video_id="TEST", max_seconds=10)
