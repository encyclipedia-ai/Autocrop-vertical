import sys
import time
import subprocess
import argparse
import os

# --- Constants ---
ASPECT_RATIO = 9 / 16

# Lazy-loaded models — initialized on first use so that importing the module
# or running --help doesn't trigger heavyweight model loading.
_model = None
_face_cascade = None

def get_yolo_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')
    return _model

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade

def analyze_scene_content(video_path, scene_start_time, scene_end_time):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    results = get_yolo_model()([frame], verbose=False)
    
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_box = [x1, y1, x2, y2]
                
                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = get_face_cascade().detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                face_box = None
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({'person_box': person_box, 'face_box': face_box})
                
    cap.release()
    return detected_objects


def detect_scenes(video_path):
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    return scene_list, fps

def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]

def decide_cropping_strategy(scene_analysis, frame_height):
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * ASPECT_RATIO
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None

def calculate_crop_box(target_box, frame_width, frame_height):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * ASPECT_RATIO)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2

def get_video_properties(video_path):
    """Returns (width, height, fps) from OpenCV — the same backend that reads frames."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def has_audio_stream(video_path):
    """Uses ffprobe to check whether the file contains an audio stream."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        return result.returncode == 0 and 'audio' in result.stdout
    except FileNotFoundError:
        # ffprobe not available — assume audio exists and let ffmpeg handle it
        return True

def get_stream_start_time(video_path, stream_type='v:0'):
    """Returns the start_time of a stream in seconds (0.0 if unavailable)."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', stream_type,
             '-show_entries', 'stream=start_time', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0.0

def is_variable_frame_rate(video_path):
    """Uses ffprobe to check if the video has a variable frame rate."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        # ffprobe returns "num/den" for both r_frame_rate and avg_frame_rate
        parts = result.stdout.strip().split(',')
        if len(parts) < 2:
            return False
        def parse_rate(s):
            nums = s.strip().split('/')
            if len(nums) == 2 and int(nums[1]) != 0:
                return int(nums[0]) / int(nums[1])
            return float(nums[0])
        r_fps = parse_rate(parts[0])
        avg_fps = parse_rate(parts[1])
        # If the real frame rate and average frame rate differ significantly, it's VFR
        return abs(r_fps - avg_fps) > 0.5
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        return False

def normalize_to_cfr(video_path, output_path):
    """Re-muxes a VFR video to constant frame rate."""
    print("  Normalizing variable frame rate to constant frame rate...")
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vsync', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Warning: VFR normalization failed, proceeding with original file.")
        print("  Stderr:", e.stderr.decode())
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smartly crops a horizontal video into a vertical one.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output video file.")
    args = parser.parse_args()

    # Defer heavy imports until after arg parsing so --help is instant
    import cv2
    import numpy as np
    from tqdm import tqdm

    script_start_time = time.time()

    input_video = args.input
    final_output_video = args.output

    # Ensure the output filename has a video extension so FFmpeg can determine the format
    _, ext = os.path.splitext(final_output_video)
    if not ext:
        final_output_video += '.mp4'
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.mkv"
    temp_cfr_input = f"{base_name}_temp_cfr_input.mp4"
    
    def cleanup_temp_files():
        """Remove any leftover temporary files."""
        for f in [temp_video_output, temp_audio_output, temp_cfr_input]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # Clean up previous temp files if they exist
    cleanup_temp_files()
    if os.path.exists(final_output_video): os.remove(final_output_video)

    # Pre-processing: normalize VFR to CFR if needed
    if is_variable_frame_rate(input_video):
        print("⚠️  Variable frame rate detected — normalizing to constant frame rate first...")
        if normalize_to_cfr(input_video, temp_cfr_input):
            input_video = temp_cfr_input
            print("✅ VFR normalization complete.")
        else:
            print("⚠️  Proceeding with original VFR file (audio sync may be affected).")

    print("🎬 Step 1: Detecting scenes...")
    step_start_time = time.time()
    scenes, _ = detect_scenes(input_video)
    step_end_time = time.time()
    
    if not scenes:
        print("❌ No scenes were detected. Aborting.")
        sys.exit(1)
    
    print(f"✅ Found {len(scenes)} scenes in {step_end_time - step_start_time:.2f}s. Here is the breakdown:")
    for i, (start, end) in enumerate(scenes):
        print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")


    print("\n🧠 Step 2: Analyzing scene content and determining strategy...")
    step_start_time = time.time()
    # Get fps from OpenCV — the same backend that reads the frames — to avoid
    # frame-rate mismatches between the reader and encoder that cause audio drift.
    original_width, original_height, fps = get_video_properties(input_video)
    
    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * ASPECT_RATIO)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    scenes_analysis = []
    for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing Scenes")):
        analysis = analyze_scene_content(input_video, start_time, end_time)
        strategy, target_box = decide_cropping_strategy(analysis, original_height)
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box
        })
    step_end_time = time.time()
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

    print("\n📋 Step 3: Generated Processing Plan")
    for i, scene_data in enumerate(scenes_analysis):
        num_people = len(scene_data['analysis'])
        strategy = scene_data['strategy']
        start_time = scenes[i][0].get_timecode()
        end_time = scenes[i][1].get_timecode()
        print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s). Strategy: {strategy}")

    print("\n✂️ Step 4: Processing video frames...")
    step_start_time = time.time()
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-r', str(fps), '-vsync', 'cfr',
        '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    dropped_frames = 0
    last_output_frame = None
    
    with tqdm(total=total_frames, desc="Applying Plan") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_scene_index < len(scenes_analysis) - 1 and \
               frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                current_scene_index += 1

            scene_data = scenes_analysis[current_scene_index]
            strategy = scene_data['strategy']
            target_box = scene_data['target_box']

            try:
                if strategy == 'TRACK':
                    crop_box = calculate_crop_box(target_box, original_width, original_height)
                    processed_frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    output_frame = cv2.resize(processed_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else: # LETTERBOX
                    scale_factor = OUTPUT_WIDTH / original_width
                    scaled_height = int(original_height * scale_factor)
                    scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height))
                    
                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
                    y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
                    output_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
                last_output_frame = output_frame
            except Exception:
                # If frame processing fails, duplicate the last good frame to
                # maintain frame count and prevent audio drift.
                dropped_frames += 1
                if last_output_frame is not None:
                    output_frame = last_output_frame
                else:
                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            
            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
    
    if dropped_frames > 0:
        print(f"  ⚠️  {dropped_frames} frame(s) could not be processed and were duplicated from the previous frame.")
    
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n❌ FFmpeg frame processing failed.")
        print("Stderr:", stderr_output)
        cleanup_temp_files()
        sys.exit(1)
    step_end_time = time.time()
    print(f"✅ Video processing complete in {step_end_time - step_start_time:.2f}s.")

    input_has_audio = has_audio_stream(input_video)

    if input_has_audio:
        print("\n🔊 Step 5: Extracting original audio...")
        step_start_time = time.time()

        # Some files have a non-zero video start_time (e.g. audio starts at 0s
        # but video starts at 1.8s). OpenCV ignores this offset and reads frames
        # from the first video frame, so the processed video starts at 0s.
        # We must trim the audio to match: skip audio before the video started,
        # and limit to the video's duration.
        video_start = get_stream_start_time(input_video, 'v:0')
        audio_extract_command = [
            'ffmpeg', '-y', '-ss', str(video_start),
            '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
        ]
        try:
            subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Audio extracted in {step_end_time - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Audio extraction failed.")
            print("Stderr:", e.stderr.decode())
            cleanup_temp_files()
            sys.exit(1)

        print("\n✨ Step 6: Merging video and audio...")
        step_start_time = time.time()
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', '-shortest', final_output_video
        ]
        try:
            subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Final video merged in {step_end_time - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Final merge failed.")
            print("Stderr:", e.stderr.decode())
            cleanup_temp_files()
            sys.exit(1)

        cleanup_temp_files()
    else:
        print("\n🔇 Step 5: No audio stream detected, skipping audio extraction.")
        # Just rename the temp video as the final output
        os.rename(temp_video_output, final_output_video)
        cleanup_temp_files()

    script_end_time = time.time()
    print(f"\n🎉 All done! Final video saved to {final_output_video}")
    print(f"⏱️  Total execution time: {script_end_time - script_start_time:.2f} seconds.")
