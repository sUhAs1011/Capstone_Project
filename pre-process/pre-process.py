import cv2
import os
import numpy as np

# Input folder containing crime videos
video_folder = r"C:\Users\Admin\Downloads\pre-process\original_videos"

# Output folder to save motion vectors
output_folder = r"C:\Users\Admin\Downloads\pre-process\optical_flow"
os.makedirs(output_folder, exist_ok=True)

# List video files with supported formats
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Target resolution for downscaling
target_width, target_height = 640, 480

# Normalized FPS
normalized_fps = 30

# Process each video
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error opening {video_file}, skipping...")
        continue

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Processing {video_file}: Original FPS={original_fps}, Normalized FPS={normalized_fps}")

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"❌ Error reading first frame of {video_file}")
        cap.release()
        continue

    # Resize and convert to grayscale
    prev_frame = cv2.resize(prev_frame, (target_width, target_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_vectors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        motion_vectors.append(flow)
        prev_gray = gray

    cap.release()

    # Save compressed motion vectors as .npz and include FPS
    save_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_flow.npz")
    np.savez_compressed(save_path,
                        motion_vectors=np.array(motion_vectors, dtype=np.float32),
                        fps=normalized_fps)

    print(f"✅ Saved Optical Flow data for {video_file} at {save_path}")

print("🎉 All videos processed with normalized FPS = 30.")
