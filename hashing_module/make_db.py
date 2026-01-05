import sqlite3
from config.config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table with both cryptographic and perceptual hash columns
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS frame_hashes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT,
        video_id TEXT,
        frame_number INTEGER,
        timestamp REAL,
        hash TEXT,          -- cryptographic hash (SHA256/BLAKE2b)
        phash TEXT,         -- perceptual hash (pHash for replay detection)
        hash_algo TEXT,
        gray INTEGER,
        width INTEGER,
        height INTEGER,
        serialize_mode TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()
    print(f"[INFO] Initialized DB at {DB_PATH}")
