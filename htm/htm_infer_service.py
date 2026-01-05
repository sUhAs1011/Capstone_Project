#!/usr/bin/env python3
import sys
import os
import json
import time
import csv
import numpy as np
import cv2
import warnings
from pathlib import Path
from collections import deque
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "encoder_test"))

from encoder_test.encoder import OpticalFlowEncoder
from encoder_test.project_utils import project_frame_to_indices
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory

# CONFIG
LOGS_DIR = os.path.join(project_root, "htm", "logs")
ALERTS_DIR = os.path.join(project_root, "htm", "alerts")
MODELS_DIR = os.path.join(project_root, "htm", "models") 
READY_FLAG_PATH = os.path.join(project_root, "htm_ready.flag")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

MAX_MAGNITUDE = 15.0
GRID_ROWS, GRID_COLS = 5, 5
ACTIVE_BITS, SEED = 40, 42
FRAME_THRESHOLD = 0.5
WINDOW_SIZE = 80
SUSPICIOUS_THRESHOLD_PCT = 15.0
PANIC_THRESHOLD_PCT = 25.0
HISTORY_AVG_THRESHOLD = 0.30
HTM_WARMUP_FRAMES = 0


class AlertManager:
    def __init__(self, video_id):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(ALERTS_DIR, f"alerts_events_{ts}.jsonl")
        self.in_attack = False
        self.last_log_time = 0
        self.start_time = None
        self.video_id = video_id
        
        with open(self.path, "w") as f: f.write("")
        print(f"[HTM] Alert Log: {self.path}", flush=True)

    def update(self, frame_idx, win_rate, run_avg, anom):
        status = "NORMAL"
        reason = "-"
        panic = win_rate >= PANIC_THRESHOLD_PCT
        corrob = (win_rate >= SUSPICIOUS_THRESHOLD_PCT) and (run_avg >= HISTORY_AVG_THRESHOLD)
        
        if panic: status, reason = "CRITICAL", f"PANIC: Window {win_rate:.1f}%"
        elif corrob: status, reason = "WARNING", f"CONFIRMED: Win {win_rate:.1f}% + Avg {run_avg:.2f}"
        
        active = status in ["CRITICAL", "WARNING"]
        now = time.time()
        
        payload = {
            "timestamp": "", 
            "frame": frame_idx, 
            "video_id": self.video_id, # LOG THE SESSION ID
            "event": "", 
            "status": status, 
            "reason": reason
        }
        
        if active and not self.in_attack:
            self.in_attack = True
            self.start_time = datetime.utcnow()
            self.last_log_time = now
            payload.update({"timestamp": self.start_time.isoformat(), "event": "ATTACK_START"})
            self._log(payload)
            
        elif active and self.in_attack:
            if now - self.last_log_time > 2.0: 
                self.last_log_time = now
                payload.update({"timestamp": datetime.utcnow().isoformat(), "event": "ATTACK_UPDATE"})
                self._log(payload)

        elif not active and self.in_attack:
            if win_rate < SUSPICIOUS_THRESHOLD_PCT:
                self.in_attack = False
                payload.update({"timestamp": datetime.utcnow().isoformat(), "event": "ATTACK_END"})
                self._log(payload)
        
        return self.in_attack, status, reason

    def _log(self, data):
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()

class RealTimeStream:
    def __init__(self, src):
        if str(src).startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024"
        print(f"[HTM] Connecting to {src}...", flush=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {src}")
        print("[HTM] Connected!", flush=True)
        self.idx = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps): self.fps = 30.0

    def read_fresh(self, elapsed_time):
        frames_to_skip = int(elapsed_time * self.fps)
        frames_to_skip = max(1, min(frames_to_skip, 100))
        for _ in range(frames_to_skip):
            self.cap.grab()
        ret, frame = self.cap.read()
        if ret:
            self.idx += 1
            return cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        return None

    def release(self): self.cap.release()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", required=True, help="Session ID from backend")
    parser.add_argument("--model-dir", default=MODELS_DIR)
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOGS_DIR, f"inference_stream_{ts}.txt")
    print(f"[HTM] Raw Log: {log_path}", flush=True)
    
    with open(log_path, "w") as f:
        f.write("Frame | Anom | Avg | WinRate | Status\n")
        f.flush()

    try:
        import pickle
        with open(os.path.join(args.model_dir, "model_meta.json"), "r") as f: meta = json.load(f)
        inp_size = meta.get("input_size", 2048)
        with open(os.path.join(args.model_dir, "sp_model.pkl"), "rb") as f: sp = pickle.load(f)
        with open(os.path.join(args.model_dir, "tm_model.pkl"), "rb") as f: tm = pickle.load(f)
        tm.reset()
    except Exception as e:
        print(f"[HTM] Model Error: {e}", flush=True)
        return

    stream = RealTimeStream(args.video)
    flow_q = deque(maxlen=2)
    win_q = deque(maxlen=WINDOW_SIZE)
    encoder = OpticalFlowEncoder(max_magnitude=MAX_MAGNITUDE)
    alert_mgr = AlertManager(args.video_id)
    
    run_sum = 0.0
    cnt = 0
    prev_duration = 0.0
    flag_created = False
    
    with open(log_path, "a") as logfile:
        try:
            while True:
                loop_start = time.time()
                frame = stream.read_fresh(prev_duration)
                if frame is None:
                    print("[HTM] Stream read failed (Retrying...)", flush=True)
                    time.sleep(0.1)
                    continue 
                
                if not flag_created:
                    with open(READY_FLAG_PATH, "w") as f: f.write("ready")
                    print(f"[HTM] Handshake Signal sent: {READY_FLAG_PATH}", flush=True)
                    flag_created = True
                
                flow_q.append(frame)
                if len(flow_q) < 2: 
                    prev_duration = time.time() - loop_start
                    continue
                
                flow = cv2.calcOpticalFlowFarneback(flow_q[-2], flow_q[-1], None, 0.5, 3, 15, 2, 5, 1.2, 0)
                
                h, w = flow.shape[:2]
                ch, cw = h//GRID_ROWS, w//GRID_COLS
                cell_sdrs = []
                for r in range(GRID_ROWS):
                    for c in range(GRID_COLS):
                        patch = flow[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
                        sdr = encoder.encode(patch)
                        cell_sdrs.append(np.array(sdr.sparse, dtype=np.int32) if getattr(sdr, "sparse", None) is not None else np.array([], dtype=np.int32))
                
                g_idx = project_frame_to_indices(cell_sdrs, inp_size, GRID_ROWS*GRID_COLS, int(np.ceil(ACTIVE_BITS/25)), int(np.ceil(inp_size/25)), SEED, ACTIVE_BITS, 0)
                sdr_in = SDR(inp_size)
                sdr_in.sparse = np.array(sorted(g_idx), dtype=np.int32)
                
                ac = SDR(sp.getColumnDimensions())
                sp.compute(sdr_in, learn=False, output=ac)
                tm.compute(ac, learn=False)
                anom = tm.anomaly
                
                cnt += 1
                run_sum += anom
                avg = run_sum / cnt
                
                if cnt < HTM_WARMUP_FRAMES:
                    status = "WARMUP"
                    rate = 0.0
                    line = f"{stream.idx:<6} | {anom:.4f} | {avg:.3f} | {rate:5.1f}% | {status}\n"
                    logfile.write(line)
                    logfile.flush()
                    if cnt % 10 == 0: print(f"\r[HTM] Warming up... {cnt}/{HTM_WARMUP_FRAMES}", end="")
                    prev_duration = time.time() - loop_start
                    continue
                
                win_q.append(anom >= FRAME_THRESHOLD)
                rate = (sum(win_q)/len(win_q))*100 if win_q else 0.0
                
                is_active, status, reason = alert_mgr.update(stream.idx, rate, avg, anom)
                
                line = f"{stream.idx:<6} | {anom:.4f} | {avg:.3f} | {rate:5.1f}% | {status}\n"
                logfile.write(line)
                logfile.flush()
                
                if stream.idx % 10 == 0:
                    print(f"\r[HTM] Fr:{stream.idx} A:{anom:.2f} W:{rate:.1f}%", end="")
                
                prev_duration = time.time() - loop_start

        except KeyboardInterrupt:
            pass
        finally:
            stream.release()

if __name__ == "__main__":
    main()
