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
*   **High Performance:** Video encoding runs entirely inside FFmpeg via a native `filter_complex` pipeline — Python never touches pixel data. A 12-minute 1080p video processes in ~51 seconds on an M1 Mac.

---

### Changelog

#### v1.3.0 (2026-02-15) — Native FFmpeg Pipeline

**Performance:**

*   **3x faster video encoding via native FFmpeg `filter_complex`.** The entire frame processing pipeline has been moved out of Python and into a single FFmpeg command. Instead of decoding every frame into Python, manipulating it with numpy/OpenCV, and piping raw bytes back to FFmpeg, the script now builds an FFmpeg filtergraph (`trim` → `crop`/`scale`/`pad` → `concat`) and lets FFmpeg handle decode, filter, and encode in C — with zero-copy frame passing between stages. Python never touches pixel data during encoding.

    Benchmarks on Apple M1 (MacBook Pro):

    | Input | Resolution | Duration | v1.2 (Python loop) | v1.3 (native FFmpeg) | Speedup |
    |-------|-----------|----------|--------------------|--------------------|---------|
    | Conan clip | 1280x720 | 49s | 3.95s | **0.97s** | **4.1x** |
    | Podcast interview | 1920x1080 | 12 min | 62.8s | **21.1s** | **3.0x** |

    End-to-end (including scene detection + YOLO analysis):

    | Input | v1.2 total | v1.3 total | Speed |
    |-------|-----------|-----------|-------|
    | Conan clip (720p, 49s) | ~12s | **~6s** | 8.3x real-time |
    | Podcast (1080p, 12 min) | 93.5s | **51.3s** | 13.7x real-time |

    Scene detection (PySceneDetect) is now the dominant bottleneck at ~50% of total time. Encoding is no longer the limiting factor.

**New Features:**

*   **Configurable aspect ratio (`--ratio`).** Output is no longer locked to 9:16. Use `--ratio 4:5` for Instagram feed, `--ratio 1:1` for square, or any custom W:H ratio.
*   **Quality presets (`--quality`).** Choose between `fast` (CRF 28, veryfast), `balanced` (CRF 23, fast — default), or `high` (CRF 18, slow). Power users can override directly with `--crf` and `--preset`.
*   **Dry-run mode (`--plan-only`).** Runs scene detection and analysis only, prints the processing plan, and exits without encoding. Useful for previewing decisions before committing to a long encode.
*   **Fixed output pixel format.** Encoder now outputs `yuv420p` instead of `yuv444p`, which is compatible with all players and platforms and produces smaller files.
*   **Improved logging and progress reporting.** Input file summary upfront (resolution, duration, fps, codec, file size, frame count), progress bars on all slow operations, and a final summary with output size, compression ratio, and processing speed.

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
    2.  `PySceneDetect` scans the video and returns a list of scene timestamps.
    3.  For each scene, `OpenCV` extracts a sample frame and `YOLOv8` detects people in it.
    4.  A set of rules determines the strategy (`TRACK` or `LETTERBOX`) for each scene based on the number and position of detected people.
    5.  The script builds an FFmpeg `filter_complex` graph — one `trim→crop→scale` or `trim→scale→pad` chain per scene, concatenated together — and executes it as a single FFmpeg command. Python never touches pixel data; FFmpeg handles decode, filter, and encode entirely in C.
    6.  Audio is extracted separately (with start-time offset correction), then merged with the processed video.

*   **Performance & Optimizations:**
    Since v1.3, the encoding pipeline runs entirely inside FFmpeg using a `filter_complex` graph. Python only acts as the "planner" (scene detection + YOLO analysis), then hands off a filtergraph to FFmpeg for execution. This eliminates all Python overhead from the hot path: no GIL contention, no BGR↔YUV conversion, no per-frame memory allocation, no 6MB-per-frame pipe I/O.

    *   **Encoding benchmarks (Apple M1):** 720p encodes at ~50x real-time, 1080p at ~30x real-time.
    *   **End-to-end:** A 12-minute 1080p video completes in ~51 seconds (13.7x real-time). Scene detection is now the dominant bottleneck.

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

    ```bash
    # Basic usage (9:16, balanced quality)
    python3 main.py -i video.mp4 -o vertical.mp4

    # Instagram feed (4:5) with high quality
    python3 main.py -i video.mp4 -o vertical.mp4 --ratio 4:5 --quality high

    # Preview the processing plan without encoding
    python3 main.py -i video.mp4 -o vertical.mp4 --plan-only

    # Full control over encoding
    python3 main.py -i video.mp4 -o vertical.mp4 --crf 20 --preset medium
    ```

---

### Prerequisites

*   Python 3.8+
*   **FFmpeg:** This script requires `ffmpeg` and `ffprobe` to be installed and available in your system's PATH. They can be installed via a package manager (e.g., `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian/Ubuntu).
