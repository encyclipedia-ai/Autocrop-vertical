# AutoCrop-Vertical: A Smart Video Cropper for Social Media (Horizontal -> Vertical)

![Demo of AutoCrop-Vertical](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

AutoCrop-Vertical is a Python script that automatically converts horizontal videos into a vertical format suitable for platforms like TikTok, Instagram Reels, and YouTube Shorts.

Instead of a simple, static center crop, this script analyzes video content scene-by-scene. It uses object detection to locate people and decides whether to tightly crop the frame on the subjects or to apply letterboxing to preserve a wide shot's composition.

---

### Key Features

*   **Content-Aware Cropping:** Uses a YOLOv8 model to detect people and automatically centers the vertical frame on them.
*   **Automatic Letterboxing:** If multiple people are too far apart for a vertical crop, the script automatically adds black bars (letterboxing) to show the full scene.
*   **Scene-by-Scene Processing:** All decisions are made per-scene, ensuring a consistent and logical edit without jarring transitions.
*   **Native Resolution:** The output resolution is dynamically calculated based on the source video's height to prevent quality loss from unnecessary upscaling.
*   **High Performance:** The processing is offloaded to FFmpeg via a direct pipe, resulting in extremely fast encoding and low CPU usage.

---

### Changelog

#### v1.1.0 (2026-02-14)

**Bug Fixes:**

*   **Fixed audio/video desynchronization.** This was caused by two separate issues:
    *   The frame rate was being read from PySceneDetect while frames were read by OpenCV. A mismatch between the two (e.g. 29.97 vs 30.0) caused the encoded video duration to drift from the audio. FPS is now read from OpenCV (the same backend that reads the frames) with explicit `-vsync cfr` enforcement.
    *   Many source files (especially YouTube downloads) have a non-zero `start_time` on the video stream (e.g. audio at 0.0s, video at 1.8s). The script now detects this offset via `ffprobe` and trims the extracted audio to match, so the two streams stay aligned.
*   **Fixed crash on videos without an audio stream.** The script now detects whether audio exists using `ffprobe` and skips the audio extraction/merge steps gracefully.
*   **Fixed hardcoded `.aac` temp audio file.** The temp audio container is now `.mkv`, which accepts any audio codec. Previously, source files with non-AAC audio (MP3, Opus, AC3, etc.) could fail or produce corrupt output.
*   **Fixed crash when output path has no file extension.** The script now auto-appends `.mp4` if no extension is provided.
*   **Fixed orphaned temp files on failure.** Temporary files are now cleaned up on all exit paths, not just on success.

**Improvements:**

*   **Variable frame rate (VFR) handling.** Phone-recorded videos often use VFR, which caused frame timing drift. The script now detects VFR sources via `ffprobe` and normalizes them to constant frame rate before processing.
*   **Corrupt frame resilience.** If a frame fails to process (bad crop, corrupt data), it is duplicated from the previous good frame instead of being dropped. This preserves the total frame count and prevents audio drift.
*   **Lazy model loading.** YOLO and Haar cascade models are now loaded on first use instead of at import time. Heavy library imports (`torch`, `ultralytics`, `cv2`, etc.) are deferred until after argument parsing, so `--help` is instant.
*   **Pinned dependency versions.** `requirements.txt` now specifies compatible version ranges to prevent breakage from upstream changes.
*   **Replaced `exit()` with `sys.exit(1)`.** Ensures proper exit codes and reliable behavior in all environments.

---

### Technical Details

This script is built on a pipeline that uses specialized libraries for each step:

*   **Core Libraries:**
    *   `PySceneDetect`: For accurate, content-aware scene cut detection.
    *   `Ultralytics (YOLOv8)`: For fast and reliable person detection.
    *   `OpenCV`: Used for frame manipulation, face detection (as a fallback), and reading video properties.
    *   `FFmpeg` / `ffprobe`: The backbone of video encoding, audio extraction, and media stream analysis.
    *   `tqdm`: For clean and informative progress bars in the console.

*   **Processing Pipeline:**
    1.  **(Pre-processing)** If the source is VFR, it is normalized to constant frame rate.
    2.  The script uses `PySceneDetect` to get a list of all scene timestamps.
    3.  It then loops through each scene and uses `OpenCV` to extract a sample frame.
    4.  This frame is passed to a pre-trained `yolov8n.pt` model to get bounding boxes for all detected people.
    5.  A set of rules determines the strategy (`TRACK` or `LETTERBOX`) for each scene based on the number and position of the detected people.
    6.  The script re-reads the input video, applies the planned transformation to every frame, and pipes the raw `bgr24` pixel data to an `FFmpeg` subprocess for efficient encoding.
    7.  Audio is extracted separately (with start-time offset correction), then merged with the processed video.

*   **Performance & Optimizations:**
    The main performance gain comes from avoiding slow, frame-by-frame processing within a pure Python loop for *writing* the video. By piping frames directly to FFmpeg's optimized C-based `libx264` encoder, we achieve significant speed.

    *   **Example Benchmark:** On a test 5-minute, 640x360 source video, the entire analysis and conversion process completes in **~11 seconds** on an Apple M1 processor. The video encoding itself runs at over 70x real-time speed.

---

### Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kamilstanuch/AutoCrop-Vertical.git
    cd AutoCrop-Vertical
    ```

2.  **Set up the environment:**
    A Python virtual environment is recommended.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    The `yolov8n.pt` model weights will be downloaded automatically on the first run.

3.  **Run the script:**
    Use the `--input` and `--output` arguments to specify the source and destination files.

    ```bash
    python main.py --input path/to/horizontal_video.mp4 --output path/to/vertical_video.mp4
    ```

---

### Prerequisites

*   Python 3.8+
*   **FFmpeg:** This script requires `ffmpeg` and `ffprobe` to be installed and available in your system's PATH. They can be installed via a package manager (e.g., `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian/Ubuntu).
