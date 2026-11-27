import os
import tempfile
import unittest

import numpy as np

import main as autocrop
import speaker

FPS = 30


def scores(length, **turns):
    """Build {track_id: array} from turns like a=[(0, 90)], b=[(90, 165)]."""
    out = {}
    for name, spans in turns.items():
        arr = np.zeros(length)
        for start, end, *value in spans:
            arr[start:end] = value[0] if value else 0.9
        out[int(name[1:])] = arr
    return out


class SpeakerTurnSegmentation(unittest.TestCase):
    def test_single_speaker_is_one_segment(self):
        segs = speaker.segment_speaker_turns(
            scores(240, t0=[(0, 240)], t1=[]), 100, 340, FPS)
        self.assertEqual(segs, [{"start_frame": 100, "end_frame": 340,
                                 "speaker": 0, "confidence": 0.9}])

    def test_interjection_shorter_than_dwell_does_not_switch(self):
        segs = speaker.segment_speaker_turns(
            scores(240, t0=[(0, 240)], t1=[(100, 120, 0.99)]), 0, 240, FPS,
            min_dwell_sec=1.2)
        self.assertEqual([s["speaker"] for s in segs], [0])

    def test_turn_longer_than_dwell_switches_at_turn_start(self):
        segs = speaker.segment_speaker_turns(
            scores(240, t0=[(0, 120)], t1=[(120, 240)]), 0, 240, FPS,
            min_dwell_sec=1.0, smooth_sec=0)
        self.assertEqual([s["speaker"] for s in segs], [0, 1])
        self.assertEqual(segs[0]["end_frame"], 120)
        self.assertEqual(segs[1]["start_frame"], 120)
        self.assertEqual(segs[-1]["end_frame"], 240)

    def test_sustained_crosstalk_becomes_group_with_group_policy(self):
        segs = speaker.segment_speaker_turns(
            scores(300, t0=[(0, 100), (100, 200, 0.8), (200, 300)],
                   t1=[(100, 200, 0.85)]), 0, 300, FPS,
            min_dwell_sec=1.0, smooth_sec=0, overlap="group")
        self.assertEqual([s["speaker"] for s in segs], [0, "group", 0])

    def test_crosstalk_follows_the_louder_face_by_default(self):
        segs = speaker.segment_speaker_turns(
            scores(300, t0=[(0, 100), (100, 200, 0.8), (200, 300)],
                   t1=[(100, 200, 0.85)]), 0, 300, FPS,
            min_dwell_sec=1.0, smooth_sec=0)
        self.assertEqual([s["speaker"] for s in segs], [0, 1, 0])
        self.assertEqual(segs[1]["start_frame"], 100)

    def test_flapping_crosstalk_does_not_move_the_frame(self):
        # Scores trade the lead every 5 frames; no candidate persists for the
        # dwell, so the frame stays on the original speaker.
        a = np.full(300, 0.9)
        b = np.zeros(300)
        for k in range(100, 200, 10):
            b[k:k + 5] = 0.95
        segs = speaker.segment_speaker_turns({0: a, 1: b}, 0, 300, FPS,
                                             min_dwell_sec=1.0, smooth_sec=0)
        self.assertEqual([s["speaker"] for s in segs], [0])

    def test_silence_holds_current_speaker(self):
        segs = speaker.segment_speaker_turns(
            scores(300, t0=[(0, 60)], t1=[(200, 300)]), 0, 300, FPS,
            min_dwell_sec=1.0, smooth_sec=0)
        self.assertEqual([s["speaker"] for s in segs], [0, 1])
        self.assertEqual(segs[1]["start_frame"], 200)

    def test_nobody_speaking_is_a_single_group_segment(self):
        segs = speaker.segment_speaker_turns(
            scores(90, t0=[], t1=[]), 0, 90, FPS)
        self.assertEqual(segs, [{"start_frame": 0, "end_frame": 90,
                                 "speaker": "group", "confidence": 0.0}])

    def test_segments_tile_the_scene_exactly(self):
        segs = speaker.segment_speaker_turns(
            scores(400, t0=[(0, 100), (250, 400)], t1=[(100, 250)]), 50, 450, FPS,
            min_dwell_sec=1.0, smooth_sec=0.2)
        self.assertEqual(segs[0]["start_frame"], 50)
        self.assertEqual(segs[-1]["end_frame"], 450)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a["end_frame"], b["start_frame"])


class FaceTracking(unittest.TestCase):
    def test_associate_continues_tracks_and_opens_new_ones(self):
        tracks = []
        speaker.associate_tracks(tracks, [[100, 100, 160, 160], [800, 100, 860, 160]], 0, 15)
        speaker.associate_tracks(tracks, [[104, 101, 164, 161], [796, 100, 856, 160]], 2, 15)
        speaker.associate_tracks(tracks, [[108, 102, 168, 162]], 4, 15)
        speaker.associate_tracks(tracks, [[110, 102, 170, 162], [500, 300, 540, 340]], 6, 15)
        self.assertEqual(len(tracks), 3)
        self.assertEqual(tracks[0]["last"], 6)
        self.assertEqual(tracks[1]["last"], 2)
        self.assertEqual(tracks[2]["first"], 6)

    def test_gap_longer_than_max_gap_starts_a_new_track(self):
        tracks = []
        speaker.associate_tracks(tracks, [[100, 100, 160, 160]], 0, max_gap_frames=4)
        speaker.associate_tracks(tracks, [[100, 100, 160, 160]], 10, max_gap_frames=4)
        self.assertEqual(len(tracks), 2)

    def test_interpolate_fills_every_frame_linearly(self):
        track = {"id": 0, "first": 0, "last": 4,
                 "boxes": {0: [0, 0, 10, 10], 4: [40, 0, 50, 10]}}
        speaker.interpolate_track(track, 0, 5)
        self.assertEqual(sorted(track["boxes"]), [0, 1, 2, 3, 4])
        self.assertEqual(track["boxes"][2], [20, 0, 30, 10])

    def test_track_faces_drops_short_tracks_and_reads_the_scene_only(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180))
            for _ in range(90):
                writer.write(np.zeros((180, 320, 3), np.uint8))
            writer.release()
            calls = []

            def fake_detect(frame):
                n = len(calls)
                calls.append(n)
                boxes = [[20, 20, 60, 60], [200, 20, 240, 60]]
                if n < 3:  # spurious 3-detection blip
                    boxes.append([150, 120, 170, 140])
                return boxes

            tracks = speaker.track_faces(path, 30, 90, FPS, face_stride=2,
                                         min_track_sec=0.5, detect=fake_detect)
        self.assertEqual(len(calls), 30)  # 60 frames / stride 2
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            self.assertEqual(min(track["boxes"]), 30)
            self.assertEqual(max(track["boxes"]), 88)
        self.assertEqual(speaker.median_box(tracks[1], 30, 90), [200, 20, 240, 60])

    def test_face_stride_follows_the_frame_rate(self):
        # 2 is the step at 30 fps. Faster clips step further, slower keep it.
        self.assertEqual(speaker.face_detect_stride(30, 2), 2)
        self.assertEqual(speaker.face_detect_stride(29.97, 2), 2)
        self.assertEqual(speaker.face_detect_stride(24, 2), 2)
        self.assertEqual(speaker.face_detect_stride(50, 2), 3)
        self.assertEqual(speaker.face_detect_stride(60, 2), 4)
        self.assertEqual(speaker.face_detect_stride(59.94, 2), 4)
        self.assertEqual(speaker.face_detect_stride(60, 1), 2)
        self.assertEqual(speaker.face_detect_stride(60, 0), 2)
        self.assertEqual(speaker.face_detect_stride(0, 2), 2)

    def test_a_60fps_scene_is_detected_as_often_per_second_as_30fps(self):
        import cv2
        fps = 60
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
            for _ in range(120):
                writer.write(np.zeros((180, 320, 3), np.uint8))
            writer.release()
            calls = []

            def fake_detect(frame):
                calls.append(len(calls))
                return [[20, 20, 60, 60]]

            tracks = speaker.track_faces(path, 0, 120, fps, face_stride=2,
                                         detect=fake_detect)
        self.assertEqual(len(calls), 30)  # 2 s at 15 detections per second
        self.assertEqual(len(tracks), 1)
        # Frames between detections are still filled in.
        self.assertEqual(sorted(tracks[0]["boxes"]), list(range(0, 117)))

    def test_a_scene_too_short_for_any_track_is_not_read(self):
        def fail(frame):
            raise AssertionError("no face detection should run")

        # 15 frames at 30 fps is 0.5 s; a track has to last 0.6 s to be kept.
        tracks = speaker.track_faces("missing.mp4", 100, 115, FPS, detect=fail)
        self.assertEqual(tracks, [])


class SceneSplitting(unittest.TestCase):
    def scene(self):
        return {
            "start_frame": 0, "end_frame": 240,
            "start_seconds": 0.0, "end_seconds": 8.0,
            "analysis": [
                {"person_box": [220, 100, 420, 700], "face_box": None, "motion": 0.0},
                {"person_box": [860, 100, 1060, 700], "face_box": None, "motion": 0.0},
            ],
            "strategy": "LETTERBOX", "target_box": None,
        }

    def tracks(self):
        return [
            {"id": 0, "first": 0, "last": 239,
             "boxes": {n: [290, 160, 350, 220] for n in range(240)}},
            {"id": 1, "first": 0, "last": 239,
             "boxes": {n: [930, 160, 990, 220] for n in range(240)}},
        ]

    def test_split_produces_track_and_group_subscenes(self):
        segs = [
            {"start_frame": 0, "end_frame": 90, "speaker": 0, "confidence": 0.8},
            {"start_frame": 90, "end_frame": 165, "speaker": 1, "confidence": 0.7},
            {"start_frame": 165, "end_frame": 240, "speaker": "group", "confidence": 0.5},
        ]
        subs = speaker.split_scene_by_speaker(
            self.scene(), segs, self.tracks(), 720,
            lambda analysis, h: autocrop.decide_cropping_strategy(analysis, h))
        # Group segment uses decide_cropping_strategy defaults (group_framing=fill):
        # two wide people cannot fit in 9:16, so fill tracks the densest subject
        # instead of letterboxing.
        self.assertEqual([s["strategy"] for s in subs], ["TRACK", "TRACK", "TRACK"])
        self.assertEqual(subs[0]["target_box"], [290, 160, 350, 220])
        self.assertEqual(subs[1]["target_box"], [930, 160, 990, 220])
        self.assertEqual(subs[2]["target_box"], [220, 100, 420, 700])
        self.assertEqual([s.get("boundary_source") for s in subs],
                         [None, "speaker-turn", "speaker-turn"])
        self.assertAlmostEqual(subs[1]["start_seconds"], 3.0)
        self.assertAlmostEqual(subs[2]["end_seconds"], 8.0)
        self.assertEqual(subs[0]["speaker"], {"kind": "track", "track_id": 0, "confidence": 0.8})
        self.assertEqual(subs[2]["speaker"]["kind"], "group")
        self.assertEqual(len(subs[0]["face_boxes"]), 90)
        self.assertNotIn("face_boxes", subs[2])

    def test_planner_follows_across_speaker_turns_instead_of_panning(self):
        segs = [
            {"start_frame": 0, "end_frame": 90, "speaker": 0, "confidence": 0.8},
            {"start_frame": 90, "end_frame": 240, "speaker": 1, "confidence": 0.7},
        ]
        subs = speaker.split_scene_by_speaker(
            self.scene(), segs, self.tracks(), 720,
            lambda analysis, h: autocrop.decide_cropping_strategy(analysis, h))
        autocrop.plan_follow_camera(subs, 1280, 720, FPS)
        autocrop.plan_pan_transitions(None, subs, 1280, 720, FPS, pan_duration=0.4)
        self.assertEqual(subs[1]["boundary_kind"], "follow")
        self.assertIsNone(subs[1]["transition"])
        summary = autocrop.summarize_pan_plan(subs)
        self.assertEqual(summary["speaker_turns"], 1)
        self.assertEqual(summary["pan"], 0)
        self.assertEqual(summary["follow"], 1)
        before = subs[0]["follow_regions"][89]
        after = subs[1]["follow_regions"][90]
        crop_w = before[2]
        max_step = crop_w / (autocrop.CAMERA_CROSS_SEC * FPS)
        self.assertLess(abs(after[0] - before[0]), max_step + 2)
        plan = autocrop.serialize_plan(subs, 1280, 720, FPS, "9:16")
        self.assertEqual(plan["scenes"][1]["boundary_source"], "speaker-turn")
        self.assertEqual(plan["scenes"][1]["speaker"]["track_id"], 1)

    def test_apply_speaker_focus_without_scorer_keeps_scenes(self):
        scene = self.scene()
        original_track_faces = speaker.track_faces
        speaker.track_faces = lambda *a, **k: self.tracks()
        try:
            out, debug = speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720,
                lambda analysis, h: autocrop.decide_cropping_strategy(analysis, h),
                log=lambda *_: None)
        finally:
            speaker.track_faces = original_track_faces
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], {"kind": "unsplit", "reason": "no-scorer", "tracks": 2})
        self.assertEqual(debug[0]["start_frame"], 0)

    def test_a_crowd_is_scored_on_the_longest_faces_only(self):
        scene = self.scene()
        tracks = [
            {"id": index, "boxes": {n: [0, 0, 1, 1] for n in range(frames)}}
            for index, frames in enumerate([5, 100, 9, 80, 8, 70, 7, 60])
        ]
        seen = []
        original = (speaker.track_faces, speaker.score_speaking)
        speaker.track_faces = lambda *a, **k: tracks

        def capture(_video, _scene, scored, _fps, log=None):
            seen.append([track["id"] for track in scored])
            return None

        speaker.score_speaking = capture
        try:
            speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720,
                lambda analysis, h: autocrop.decide_cropping_strategy(analysis, h),
                log=lambda *_: None)
        finally:
            speaker.track_faces, speaker.score_speaking = original
        self.assertEqual(seen, [[1, 3, 5, 7, 2, 4]])

    def test_fewer_faces_than_the_cap_are_all_scored(self):
        kept = speaker.tracks_worth_scoring([
            {"id": 0, "boxes": {1: [0, 0, 1, 1]}},
            {"id": 1, "boxes": {1: [0, 0, 1, 1], 2: [0, 0, 1, 1]}},
        ])
        self.assertEqual([track["id"] for track in kept], [0, 1])

    def test_apply_speaker_focus_with_scorer_splits(self):
        scene = self.scene()
        original = (speaker.track_faces, speaker.score_speaking)
        speaker.track_faces = lambda *a, **k: self.tracks()
        speaker.score_speaking = lambda *a, **k: scores(240, t0=[(0, 120)], t1=[(120, 240)])
        try:
            out, _ = speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720,
                lambda analysis, h: autocrop.decide_cropping_strategy(analysis, h),
                min_dwell_sec=1.0, log=lambda *_: None)
        finally:
            speaker.track_faces, speaker.score_speaking = original
        self.assertEqual([s["speaker"]["track_id"] for s in out], [0, 1])
        self.assertEqual(out[1]["boundary_source"], "speaker-turn")

    def test_lone_speaking_face_among_bystanders_is_tracked(self):
        # YOLO saw 2 bodies -> fill TRACK on densest body, but only one face
        # exists and it talks — promote to that face (also covers legacy LETTERBOX).
        scene = self.scene()
        scene["strategy"], scene["target_box"] = "TRACK", [860, 100, 1060, 700]
        track = self.tracks()[0]
        original = (speaker.track_faces, speaker.score_speaking)
        speaker.track_faces = lambda *a, **k: [track]
        speaker.score_speaking = lambda *a, **k: scores(240, t0=[(0, 200)])
        try:
            out, debug = speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720,
                lambda a, h: ("TRACK", [860, 100, 1060, 700]),
                log=lambda *_: None)
        finally:
            speaker.track_faces, speaker.score_speaking = original
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["strategy"], "TRACK")
        self.assertEqual(out[0]["speaker"]["reason"], "single-speaking-face")
        self.assertEqual(out[0]["target_box"], speaker.median_box(track, 0, 240))
        self.assertEqual(len(out[0]["face_boxes"]), 240)
        self.assertIn(0, out[0]["speaking_frames"])
        self.assertIsNotNone(debug[0]["scores"])

    def test_lone_silent_face_keeps_letterbox(self):
        scene = self.scene()
        scene["strategy"], scene["target_box"] = "LETTERBOX", None
        track = self.tracks()[0]
        original = (speaker.track_faces, speaker.score_speaking)
        speaker.track_faces = lambda *a, **k: [track]
        speaker.score_speaking = lambda *a, **k: scores(240, t0=[(0, 40)])
        try:
            out, _ = speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720, lambda a, h: ("LETTERBOX", None),
                log=lambda *_: None)
        finally:
            speaker.track_faces, speaker.score_speaking = original
        self.assertEqual(out[0]["strategy"], "LETTERBOX")
        self.assertEqual(out[0]["speaker"]["kind"], "unsplit")

    def test_single_person_letterbox_scenes_are_untouched(self):
        scene = self.scene()
        scene["analysis"] = scene["analysis"][:1]
        out, debug = speaker.apply_speaker_focus(
            "unused.mp4", [scene], FPS, 720, lambda a, h: ("TRACK", None))
        self.assertIs(out[0], scene)
        self.assertEqual(debug, {})

    def test_single_person_track_keeps_a_face_path(self):
        scene = self.scene()
        scene["analysis"] = scene["analysis"][:1]
        scene["strategy"] = "TRACK"
        scene["target_box"] = [220, 100, 420, 700]
        track = {"id": 0, "first": 0, "last": 239,
                 "boxes": {n: [300, 160, 360, 220] for n in range(240)}}
        original = speaker.track_faces
        speaker.track_faces = lambda *a, **k: [track]
        try:
            out, debug = speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720, lambda a, h: ("TRACK", None),
                log=lambda *_: None)
        finally:
            speaker.track_faces = original
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]["face_boxes"]), 240)
        self.assertIsNone(out[0]["speaking_frames"])
        self.assertEqual(out[0]["focus_face"], [300, 160, 360, 220])
        self.assertEqual(debug, {})

    def test_a_scene_outside_the_focus_range_is_not_tracked(self):
        early = self.scene()
        early["end_frame"] = 60
        early["end_seconds"] = 2.0
        play = self.scene()
        play["start_frame"] = 90
        play["end_frame"] = 210
        play["start_seconds"] = 3.0
        play["end_seconds"] = 7.0
        calls = []

        def fake_track(*args, **_kwargs):
            calls.append((args[1], args[2]))
            return self.tracks()

        original = speaker.track_faces
        speaker.track_faces = fake_track
        try:
            out, _ = speaker.apply_speaker_focus(
                "unused.mp4", [early, play], FPS, 720,
                lambda a, h: ("LETTERBOX", None),
                log=lambda *_: None,
                focus_start_frame=90, focus_end_frame=210)
        finally:
            speaker.track_faces = original
        self.assertEqual(out[0]["speaker"]["reason"], "outside-focus")
        self.assertEqual(out[0]["end_frame"], 60)
        self.assertEqual(calls, [(90, 210)])

    def test_handles_hold_the_crop_from_the_play_range(self):
        scene = self.scene()
        seen = {}

        def fake_track(_path, start, end, *_args, **_kwargs):
            seen["span"] = (start, end)
            boxes = {n: [290 + (0 if n < start + 60 else 640), 160,
                         350 + (0 if n < start + 60 else 640), 220]
                     for n in range(start, end)}
            return [
                {"id": 0, "first": start, "last": start + 59, "boxes": {
                    n: boxes[n] for n in range(start, start + 60)}},
                {"id": 1, "first": start + 60, "last": end - 1, "boxes": {
                    n: boxes[n] for n in range(start + 60, end)}},
            ]

        def fake_score(_path, scored, _tracks, _fps, log=None):
            seen["scored"] = (scored["start_frame"], scored["end_frame"],
                              scored["start_seconds"], scored["end_seconds"])
            length = scored["end_frame"] - scored["start_frame"]
            return scores(length, t0=[(0, length // 2)], t1=[(length // 2, length)])

        original = (speaker.track_faces, speaker.score_speaking)
        speaker.track_faces = fake_track
        speaker.score_speaking = fake_score
        try:
            out, debug = speaker.apply_speaker_focus(
                "unused.mp4", [scene], FPS, 720,
                lambda a, h: ("LETTERBOX", None),
                min_dwell_sec=1.0, log=lambda *_: None,
                focus_start_frame=90, focus_end_frame=210)
        finally:
            speaker.track_faces, speaker.score_speaking = original
        self.assertEqual(seen["span"], (90, 210))
        self.assertEqual(seen["scored"][:2], (90, 210))
        self.assertAlmostEqual(seen["scored"][2], 3.0)
        self.assertAlmostEqual(seen["scored"][3], 7.0)
        self.assertEqual(out[0]["start_frame"], 0)
        self.assertAlmostEqual(out[0]["start_seconds"], 0.0)
        self.assertEqual(out[-1]["end_frame"], 240)
        self.assertAlmostEqual(out[-1]["end_seconds"], 8.0)
        self.assertNotIn(0, out[0]["face_boxes"])
        self.assertIn(90, out[0]["face_boxes"])
        self.assertEqual(debug[0]["start_frame"], 90)
        autocrop.plan_follow_camera(out, 1280, 720, FPS)
        # The 3 s before the play range holds the first framed face.
        self.assertEqual(out[0]["follow_regions"][0], out[0]["follow_regions"][90])
        self.assertEqual(out[-1]["follow_regions"][239], out[-1]["follow_regions"][209])


class Timecode(unittest.TestCase):
    def test_format_matches_pyscenedetect(self):
        self.assertEqual(autocrop.format_timecode(0), "00:00:00.000")
        self.assertEqual(autocrop.format_timecode(3725.5), "01:02:05.500")


if __name__ == "__main__":
    unittest.main()
