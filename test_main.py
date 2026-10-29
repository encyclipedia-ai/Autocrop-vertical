import unittest

import main


def track(start, end, x0, x1):
    return {"start_frame": start, "end_frame": end, "strategy": "TRACK",
            "target_box": [x0, 0, x1, 90]}


def letterbox(start, end):
    return {"start_frame": start, "end_frame": end, "strategy": "LETTERBOX",
            "target_box": None}


class CropModeTests(unittest.TestCase):
    def _people(self, *motions):
        # Boxes are [x1, y1, x2, y2]; faces optional.
        out = []
        for i, motion in enumerate(motions):
            x = 20 + i * 40
            out.append({
                "person_box": [x, 10, x + 30, 80],
                "face_box": [x + 5, 15, x + 25, 40],
                "motion": motion,
            })
        return out

    def test_wide_always_letterboxes(self):
        strategy, box = main.decide_cropping_strategy(
            self._people(1.0), 90, crop_mode="wide", frame_width=160)
        self.assertEqual(strategy, "LETTERBOX")
        self.assertIsNone(box)

    def test_center_locks_to_horizontal_middle(self):
        main.ASPECT_RATIO = 9 / 16
        strategy, box = main.decide_cropping_strategy(
            self._people(1.0), 90, crop_mode="center", frame_width=160)
        self.assertEqual(strategy, "TRACK")
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        self.assertEqual(y1, 0)
        self.assertEqual(y2, 90)
        crop_w = x2 - x1
        self.assertEqual(x1, (160 - crop_w) // 2)

    def test_subject_tracks_most_active_even_with_low_motion(self):
        # Auto with a high threshold would letterbox; subject forces TRACK.
        strategy, box = main.decide_cropping_strategy(
            self._people(0.01, 0.02), 90, motion_threshold=0.5,
            crop_mode="subject", frame_width=160)
        self.assertEqual(strategy, "TRACK")
        self.assertEqual(box, [65, 15, 85, 40])  # face of second (more active)

    def test_auto_fills_densest_subject_when_group_is_too_wide(self):
        # Default group_framing='fill' tracks the densest subject instead of
        # letterboxing a group that cannot fit in 9:16.
        strategy, box = main.decide_cropping_strategy(
            self._people(0.01, 0.02), 90, motion_threshold=0.5,
            crop_mode="auto", frame_width=160)
        self.assertEqual(strategy, "TRACK")
        self.assertEqual(box, [65, 15, 85, 40])  # face of second (more motion)

    def test_letterbox_group_framing_preserves_legacy_bars(self):
        strategy, box = main.decide_cropping_strategy(
            self._people(0.01, 0.02), 90, motion_threshold=0.5,
            crop_mode="auto", frame_width=160, group_framing="letterbox")
        self.assertEqual(strategy, "LETTERBOX")
        self.assertIsNone(box)

    def test_compose_pulls_nearby_screen_into_target(self):
        face = [100, 20, 140, 70]
        screen = [20, 30, 90, 80]
        composed = main.compose_target_with_objects(face, [screen], frame_width=200)
        self.assertEqual(composed, [20, 20, 140, 80])

    def test_compose_ignores_far_objects(self):
        face = [10, 20, 40, 70]
        far = [170, 20, 199, 70]
        composed = main.compose_target_with_objects(face, [far], frame_width=200)
        self.assertEqual(composed, face)

    def test_fill_tracks_lone_object_when_nobody_detected(self):
        screen = [40, 20, 120, 80]
        strategy, box = main.decide_cropping_strategy(
            [], 90, crop_mode="auto", frame_width=160,
            group_framing="fill", object_boxes=[screen])
        self.assertEqual(strategy, "TRACK")
        self.assertEqual(box, screen)

    def test_plan_object_compose_widens_speaker_face(self):
        scenes = [{
            "strategy": "TRACK",
            "target_box": [100, 20, 140, 70],
            "object_boxes": [[20, 30, 90, 80]],
        }]
        n = main.plan_object_compose(scenes, frame_width=200)
        self.assertEqual(n, 1)
        self.assertEqual(scenes[0]["target_box"], [20, 20, 140, 80])

    def test_follow_camera_off_lets_speaker_turns_pan(self):
        # With face_boxes present, plan_pan_transitions used to skip pans
        # (follow camera owned the boundary). After stripping paths for
        # --follow-camera off, a speaker-turn boundary must ease.
        main.ASPECT_RATIO = 9 / 16
        left = {
            "start_frame": 0, "end_frame": 30, "strategy": "TRACK",
            "target_box": [10, 0, 60, 90],
            "face_boxes": {0: [20, 10, 40, 40]},
            "boundary_source": "start",
        }
        right = {
            "start_frame": 30, "end_frame": 60, "strategy": "TRACK",
            "target_box": [100, 0, 150, 90],
            "face_boxes": {30: [110, 10, 130, 40]},
            "boundary_source": "speaker-turn",
        }
        scenes = [left, right]
        for scene in scenes:
            scene.pop("face_boxes", None)
            scene.pop("speaking_frames", None)
            scene.pop("follow_regions", None)
        main.plan_pan_transitions(
            None, scenes, 160, 90, fps=10,
            pan_duration=0.4, zoom_duration=0.4)
        self.assertEqual(right["boundary_kind"], "pan")
        self.assertIsNotNone(right["transition"])


class PanMathTests(unittest.TestCase):
    def test_smoothstep_is_monotonic_and_clamped(self):
        values = [main.smoothstep(i / 20) for i in range(21)]
        self.assertEqual(main.smoothstep(-1), 0)
        self.assertEqual(main.smoothstep(2), 1)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 1)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_pan_starts_and_ends_at_requested_positions(self):
        positions = [
            main.interpolate_pan_x(10, 110, frame, 5)
            for frame in range(5)
        ]
        self.assertEqual(positions[0], 10)
        self.assertEqual(positions[-1], 110)
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))
        self.assertLess(positions[1] - positions[0], positions[2] - positions[1])
        self.assertGreater(positions[3] - positions[2], positions[4] - positions[3])

    def test_interpolate_pan_x_clamps_to_source_bounds(self):
        self.assertEqual(
            main.interpolate_pan_x(-20, 200, 0, 5, min_x=0, max_x=110),
            0,
        )
        self.assertEqual(
            main.interpolate_pan_x(-20, 200, 4, 5, min_x=0, max_x=110),
            110,
        )

    def test_crop_center_overhangs_up_to_the_black_cap(self):
        # A subject at the frame edge used to pin the window at 0, which
        # shoved the subject off center. The window now hangs off the source
        # until the black cap, then stops.
        left = main.calculate_crop_box_for_center(0, 160, 90, 50)
        self.assertLess(left[0], 0)
        self.assertEqual(left[2] - left[0], 50)
        self.assertAlmostEqual(-left[0], 0.38 * 50, delta=1)
        right = main.calculate_crop_box_for_center(160, 160, 90, 50)
        self.assertGreater(right[2], 160)
        self.assertAlmostEqual(right[2] - 160, 0.38 * 50, delta=1)
        # A subject the window can cover stays full-bleed and centered.
        fitted = main.calculate_crop_box_for_center(80, 160, 90, 50)
        self.assertEqual(fitted, (55, 0, 105, 90))

    def test_region_interpolation_keeps_an_overhanging_glide(self):
        # Clamping x at 0 used to hitch the camera the moment the ideal
        # window crossed the edge. The eased origin has to be allowed to
        # go negative.
        held = main.interpolate_region((-40, 0, 50, 90), (-40, 0, 50, 90), 0, 2, 160, 90)
        self.assertEqual(held, (-40, 0, 50, 90))
        mid = main.interpolate_region((10, 0, 50, 90), (-30, 0, 50, 90), 2, 5, 160, 90)
        self.assertLess(mid[0], 0)
        self.assertEqual(mid[2], 50)

    def test_region_interpolation_matches_pan_for_equal_widths(self):
        for frame in range(5):
            x, w = main.interpolate_region((10, 50), (110, 50), frame, 5, 160)
            self.assertEqual(w, 50)
            self.assertEqual(x, main.interpolate_pan_x(10, 110, frame, 5))

    def test_region_interpolation_zooms_between_full_frame_and_crop(self):
        regions = [
            main.interpolate_region((0, 160), (110, 50), frame, 6, 160)
            for frame in range(6)
        ]
        self.assertEqual(regions[0], (0, 160))
        self.assertEqual(regions[-1], (110, 50))
        widths = [w for _, w in regions]
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])))
        for x, w in regions:
            self.assertGreaterEqual(x, -1)
            self.assertLessEqual(x + w, 161)

    def test_track_to_track_crop_jump_pans_without_reading_pixels(self):
        scenes = [track(0, 10, 10, 40), track(10, 20, 120, 150)]
        # video_path is unused for pan eligibility; production H.264 must
        # still pan when adjacent frames would fail a pixel hard-cut check.
        main.plan_pan_transitions(
            "/nonexistent.mp4", scenes, 160, 90, 10, pan_duration=0.4)

        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        transition = scenes[1]["transition"]
        self.assertIsNotNone(transition)
        self.assertEqual(transition["kind"], "pan")
        self.assertEqual(transition["from_w"], transition["to_w"])
        positions = [
            main.interpolate_pan_x(
                transition["from_x"], transition["to_x"], frame,
                transition["duration_frames"])
            for frame in range(transition["duration_frames"])
        ]
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], transition["from_x"])
        self.assertEqual(positions[-1], transition["to_x"])
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))

    def test_speaker_switch_zoom_and_jitter(self):
        scenes = [
            track(0, 10, 10, 40),
            track(10, 20, 120, 150),
            letterbox(20, 30),
        ]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4)

        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        self.assertIsNotNone(scenes[1]["transition"])
        self.assertEqual(scenes[2]["boundary_kind"], "zoom-out")
        zoom = scenes[2]["transition"]
        self.assertEqual((zoom["from_x"], zoom["from_w"]), (110, 50))
        self.assertEqual((zoom["to_x"], zoom["to_w"]), (0, 160))
        self.assertEqual(zoom["duration_frames"], 4)

        jitter_scenes = [track(0, 10, 70, 90), track(10, 20, 72, 92)]
        main.plan_pan_transitions(
            None, jitter_scenes, 160, 90, 10, pan_duration=0.4)
        self.assertEqual(jitter_scenes[1]["boundary_kind"], "hold")
        self.assertIsNone(jitter_scenes[1]["transition"])

    def test_letterbox_to_track_zooms_in(self):
        scenes = [letterbox(0, 10), track(10, 20, 120, 150)]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        self.assertEqual(scenes[1]["boundary_kind"], "zoom-in")
        zoom = scenes[1]["transition"]
        self.assertEqual((zoom["from_x"], zoom["from_w"]), (0, 160))
        self.assertEqual((zoom["to_x"], zoom["to_w"]), (110, 50))

    def test_zoom_duration_is_independent_of_pan_duration(self):
        scenes = [letterbox(0, 10), track(10, 30, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4, zoom_duration=1.0)
        self.assertEqual(scenes[1]["transition"]["duration_frames"], 10)

        scenes = [letterbox(0, 10), track(10, 30, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4, zoom_duration=0)
        self.assertEqual(scenes[1]["boundary_kind"], "layout-switch")
        self.assertIsNone(scenes[1]["transition"])

    def test_zoom_is_capped_to_scene_length(self):
        scenes = [letterbox(0, 10), track(10, 12, 120, 150)]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        self.assertEqual(scenes[1]["transition"]["duration_frames"], 2)

    def test_zero_pan_duration_disables_transitions(self):
        scenes = [track(0, 10, 10, 40), track(10, 20, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0)
        self.assertIsNone(scenes[1]["transition"])

    def test_summary_counts_transitions_against_boundaries(self):
        scenes = [
            track(0, 10, 10, 40),
            track(10, 20, 120, 150),
            track(20, 30, 121, 151),
            letterbox(30, 40),
            letterbox(40, 50),
            track(50, 60, 10, 40),
        ]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        summary = main.summarize_pan_plan(scenes)
        self.assertEqual(summary["track_to_track"], 2)
        self.assertEqual(summary["layout_boundaries"], 2)
        self.assertEqual(summary["pan"], 1)
        self.assertEqual(summary["hold"], 1)
        self.assertEqual(summary["zoom"], 2)
        self.assertEqual(summary["layout_switch"], 0)

        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4, zoom_duration=0)
        summary = main.summarize_pan_plan(scenes)
        self.assertEqual(summary["zoom"], 0)
        self.assertEqual(summary["layout_switch"], 2)


class FaceZoomTests(unittest.TestCase):
    W, H = 1280, 720

    def test_small_face_gets_a_tight_aspect_correct_region(self):
        face = [240, 128, 272, 168]          # 40px tall face in 720p
        region = main.face_zoom_region(face, self.W, self.H)
        self.assertIsNotNone(region)
        x, y, w, h = region
        # 40 / 0.18 = 222 wanted, but max upscale 2.0 caps at 360.
        self.assertEqual(h, 360)
        self.assertAlmostEqual(w / h, 9 / 16, delta=0.01)
        self.assertGreaterEqual(x, -0.38 * w - 1)
        self.assertGreaterEqual(y, -0.38 * h - 1)
        self.assertLessEqual(x + w, self.W + 0.38 * w + 1)
        self.assertLessEqual(y + h, self.H + 0.38 * h + 1)
        # Face centre sits near the vertical middle, horizontally centred.
        # This face is high in the frame, so the window overhangs the top.
        cy = (face[1] + face[3]) / 2
        self.assertAlmostEqual((cy - y) / h, main.FACE_Y_FRACTION, delta=0.03)
        self.assertLess(y, 0)
        self.assertAlmostEqual(x + w / 2, (face[0] + face[2]) / 2, delta=1)

    def test_large_face_keeps_full_height(self):
        face = [500, 200, 700, 400]          # 200px face = 28% of height
        self.assertIsNone(main.face_zoom_region(face, self.W, self.H))

    def test_marginal_gain_is_skipped(self):
        # 120px face -> 667px crop = 93% of height: not worth a zoom.
        face = [500, 200, 620, 320]
        self.assertIsNone(main.face_zoom_region(face, self.W, self.H))

    def test_disabled_or_missing_face(self):
        self.assertIsNone(main.face_zoom_region([0, 0, 30, 30], self.W, self.H, face_fraction=0))
        self.assertIsNone(main.face_zoom_region(None, self.W, self.H))

    def test_edge_face_overhangs_up_to_the_black_cap(self):
        x, y, w, h = main.face_zoom_region([0, 0, 30, 30], self.W, self.H)
        self.assertLess(x, 0)
        self.assertLess(y, 0)
        self.assertAlmostEqual(-x, main.MAX_BLACK_FRACTION * w, delta=1)
        self.assertAlmostEqual(-y, main.MAX_BLACK_FRACTION * h, delta=1)
        x, y, w, h = main.face_zoom_region([1250, 690, 1280, 720], self.W, self.H)
        self.assertGreater(x + w, self.W)
        self.assertGreater(y + h, self.H)
        self.assertAlmostEqual(x + w - self.W, main.MAX_BLACK_FRACTION * w, delta=1)
        self.assertAlmostEqual(y + h - self.H, main.MAX_BLACK_FRACTION * h, delta=1)

    def test_plan_face_zoom_only_touches_track_scenes_with_a_face(self):
        scenes = [
            {"start_frame": 0, "end_frame": 10, "strategy": "TRACK",
             "target_box": [240, 128, 272, 168], "focus_face": [240, 128, 272, 168]},
            {"start_frame": 10, "end_frame": 20, "strategy": "TRACK",
             "target_box": [600, 100, 800, 700]},                    # body box, no face
            {"start_frame": 20, "end_frame": 30, "strategy": "LETTERBOX",
             "target_box": None, "focus_face": [240, 128, 272, 168]},
        ]
        self.assertEqual(main.plan_face_zoom(scenes, self.W, self.H), 1)
        self.assertIsNotNone(scenes[0]["zoom_region"])
        self.assertIsNone(scenes[1]["zoom_region"])
        self.assertIsNone(scenes[2]["zoom_region"])
        self.assertEqual(main.summarize_pan_plan(scenes)["face_zoom"], 1)
        self.assertEqual(main.scene_steady_region(scenes[0], self.W, self.H),
                         tuple(scenes[0]["zoom_region"]))

    def test_zoom_between_full_height_and_face_crop_eases_all_four_axes(self):
        scenes = [
            {"start_frame": 0, "end_frame": 10, "strategy": "TRACK",
             "target_box": [600, 100, 800, 700]},
            {"start_frame": 10, "end_frame": 30, "strategy": "TRACK",
             "target_box": [240, 128, 272, 168], "focus_face": [240, 128, 272, 168]},
        ]
        main.plan_face_zoom(scenes, self.W, self.H)
        main.plan_pan_transitions(None, scenes, self.W, self.H, 10, pan_duration=0.5)
        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        t = scenes[1]["transition"]
        self.assertEqual((t["from_h"], t["to_h"]), (720, 360))
        regions = main.plan_frame_regions(scenes, 30, self.W, self.H)
        self.assertEqual(regions[9], (498, 0, 405, 720))   # centred on the body box
        self.assertEqual(regions[15], tuple(scenes[1]["zoom_region"]))
        heights = [r[3] for r in regions[10:15]]
        self.assertTrue(all(a >= b for a, b in zip(heights, heights[1:])))
        self.assertGreaterEqual(len(set(heights)), 3)
        for x, y, w, h in regions:
            self.assertGreaterEqual(x, -0.38 * w - 1)
            self.assertGreaterEqual(y, -0.38 * h - 1)
            self.assertLessEqual(x + w, self.W + 0.38 * w + 1)
            self.assertLessEqual(y + h, self.H + 0.38 * h + 1)

    def test_two_zoomed_faces_of_equal_size_hold_or_pan_by_position(self):
        a = {"start_frame": 0, "end_frame": 10, "strategy": "TRACK",
             "target_box": [240, 128, 272, 168], "focus_face": [240, 128, 272, 168]}
        b = {"start_frame": 10, "end_frame": 20, "strategy": "TRACK",
             "target_box": [244, 130, 276, 170], "focus_face": [244, 130, 276, 170]}
        c = {"start_frame": 20, "end_frame": 30, "strategy": "TRACK",
             "target_box": [900, 128, 932, 168], "focus_face": [900, 128, 932, 168]}
        scenes = [a, b, c]
        main.plan_face_zoom(scenes, self.W, self.H)
        main.plan_pan_transitions(None, scenes, self.W, self.H, 10, pan_duration=0.4)
        self.assertEqual(scenes[1]["boundary_kind"], "hold")
        self.assertEqual(scenes[2]["boundary_kind"], "pan")

    def test_render_region_upscales_face_crop_to_output(self):
        import numpy as np
        frame = np.zeros((self.H, self.W, 3), np.uint8)
        frame[128:168, 240:272] = 255
        out = main.render_region(frame, 55, 202, self.W, self.H, 406, 720, y=11, height=360)
        self.assertEqual(out.shape, (720, 406, 3))
        # The 40px face is now ~80px tall in the output.
        rows = np.where(out[:, :, 0].max(axis=1) > 128)[0]
        self.assertGreater(rows[-1] - rows[0], 70)


class ProductionRenderPathTests(unittest.TestCase):
    """Drive the exact per-frame functions the encode loop uses."""

    WIDTH, HEIGHT, FPS = 160, 90, 10

    def two_scene_plan(self, pan_duration=0.4):
        scenes = [track(0, 10, 10, 40), track(10, 30, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, self.WIDTH, self.HEIGHT, self.FPS,
            pan_duration=pan_duration)
        return scenes

    def zoom_plan(self, zoom_duration=0.6):
        scenes = [letterbox(0, 10), track(10, 30, 120, 150), letterbox(30, 50)]
        main.plan_pan_transitions(
            None, scenes, self.WIDTH, self.HEIGHT, self.FPS,
            pan_duration=0.4, zoom_duration=zoom_duration)
        return scenes

    def test_frame_crops_move_gradually_through_a_pan(self):
        scenes = self.two_scene_plan()
        positions = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)

        before = positions[:10]
        pan_frames = scenes[1]["transition"]["duration_frames"]
        during = positions[10:10 + pan_frames]
        after = positions[10 + pan_frames:]

        self.assertEqual(set(before), {0})
        self.assertEqual(set(after), {110})
        self.assertEqual(during[0], 0)
        self.assertEqual(during[-1], 110)
        # A real pan has several distinct intermediate positions, not a snap.
        self.assertGreaterEqual(len(set(during)), 4)
        self.assertTrue(all(a < b for a, b in zip(during, during[1:])))

    def test_frame_regions_zoom_in_and_out_gradually(self):
        scenes = self.zoom_plan()
        regions = main.plan_frame_regions(scenes, 50, self.WIDTH, self.HEIGHT)

        self.assertEqual(set(regions[:10]), {(0, 0, 160, 90)})
        zoom_in = regions[10:16]
        self.assertEqual(zoom_in[0], (0, 0, 160, 90))
        self.assertEqual(zoom_in[-1], (110, 0, 50, 90))
        widths = [r[2] for r in zoom_in]
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])))
        self.assertEqual(set(regions[16:30]), {(110, 0, 50, 90)})

        zoom_out = regions[30:36]
        self.assertEqual(zoom_out[0], (110, 0, 50, 90))
        self.assertEqual(zoom_out[-1], (0, 0, 160, 90))
        widths = [r[2] for r in zoom_out]
        self.assertTrue(all(a < b for a, b in zip(widths, widths[1:])))
        self.assertEqual(set(regions[36:]), {(0, 0, 160, 90)})

    def test_zero_pan_duration_snaps_in_one_frame(self):
        scenes = self.two_scene_plan(pan_duration=0)
        positions = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)
        self.assertEqual(positions[9], 0)
        self.assertEqual(positions[10], 110)

    def test_scene_cursor_advances_and_never_rewinds(self):
        scenes = self.two_scene_plan()
        self.assertEqual(main.scene_index_for_frame(scenes, 0, 0), 0)
        self.assertEqual(main.scene_index_for_frame(scenes, 9, 0), 0)
        self.assertEqual(main.scene_index_for_frame(scenes, 10, 0), 1)
        self.assertEqual(main.scene_index_for_frame(scenes, 5, 1), 1)

    def _column_encoded_frame(self):
        import numpy as np
        # Encode each source column's x coordinate as its pixel intensity so
        # output pixels report exactly which source columns were shown.
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(self.WIDTH, dtype=np.uint8)[None, :]
        frame[:, :, 1] = 200  # non-black so letterbox bars are detectable
        return frame

    def test_rendered_pixels_shift_gradually_across_boundary(self):
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV unavailable: {exc}")

        scenes = self.two_scene_plan()
        out_w, out_h = main.compute_output_size(self.HEIGHT)
        frame = self._column_encoded_frame()

        rendered_crop_x = []
        index = 0
        for frame_number in range(30):
            index = main.scene_index_for_frame(scenes, frame_number, index)
            out = main.render_output_frame(
                frame, scenes[index], frame_number,
                self.WIDTH, self.HEIGHT, out_w, out_h)
            self.assertEqual(out.shape, (out_h, out_w, 3))
            rendered_crop_x.append(int(out[0, 0, 0]))

        planned = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)
        self.assertEqual(rendered_crop_x, planned)
        pan_frames = scenes[1]["transition"]["duration_frames"]
        during = rendered_crop_x[10:10 + pan_frames]
        self.assertGreaterEqual(len(set(during)), 4)
        self.assertTrue(all(a < b for a, b in zip(during, during[1:])))

    def test_rendered_zoom_shrinks_letterbox_bars_gradually(self):
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV unavailable: {exc}")

        scenes = self.zoom_plan()
        out_w, out_h = main.compute_output_size(self.HEIGHT)
        frame = self._column_encoded_frame()

        bar_heights = []
        left_columns = []
        index = 0
        for frame_number in range(50):
            index = main.scene_index_for_frame(scenes, frame_number, index)
            out = main.render_output_frame(
                frame, scenes[index], frame_number,
                self.WIDTH, self.HEIGHT, out_w, out_h)
            self.assertEqual(out.shape, (out_h, out_w, 3))
            content_rows = np.where(out[:, out_w // 2, 1] > 0)[0]
            bar_heights.append(int(content_rows[0]))
            left_columns.append(int(out[out_h // 2, 0, 0]))

        # LETTERBOX steady state: bars present, whole frame visible (the
        # downscale averages the first few source columns into pixel 0).
        self.assertGreater(bar_heights[0], 0)
        self.assertLessEqual(left_columns[0], 3)
        # TRACK steady state: no bars, crop starts at x=110.
        self.assertEqual(bar_heights[20], 0)
        self.assertEqual(left_columns[20], 110)

        zoom_in = bar_heights[10:16]
        self.assertTrue(all(a >= b for a, b in zip(zoom_in, zoom_in[1:])))
        self.assertGreaterEqual(len(set(zoom_in)), 4)
        zoom_out = bar_heights[30:36]
        self.assertTrue(all(a <= b for a, b in zip(zoom_out, zoom_out[1:])))
        self.assertGreaterEqual(len(set(zoom_out)), 4)
        # The visible left edge converges on the subject instead of jumping.
        lefts = left_columns[10:16]
        self.assertTrue(all(a <= b + 2 for a, b in zip(lefts, lefts[1:])))
        self.assertGreater(lefts[-1] - lefts[0], 90)

    def test_serialized_plan_round_trips_through_json(self):
        import json
        scenes = self.zoom_plan()
        for scene in scenes:
            scene["analysis"] = [{"person_box": [1, 2, 3, 4]}]
        payload = json.loads(json.dumps(main.serialize_plan(
            scenes, self.WIDTH, self.HEIGHT, self.FPS, "9:16")))
        self.assertEqual(payload["summary"]["zoom"], 2)
        self.assertEqual(payload["scenes"][1]["boundary_kind"], "zoom-in")
        self.assertEqual(payload["scenes"][1]["transition"]["kind"], "zoom-in")
        self.assertEqual(payload["scenes"][1]["people"], 1)
        replayed = main.plan_frame_regions(
            payload["scenes"], 50, payload["width"], payload["height"])
        self.assertEqual(
            [tuple(r) for r in replayed],
            main.plan_frame_regions(scenes, 50, self.WIDTH, self.HEIGHT))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV test dependencies unavailable: {exc}")

    def test_frame_difference_separates_motion_from_hard_cut(self):
        import numpy as np

        before = np.full((90, 160, 3), 80, dtype=np.uint8)
        soft = before.copy()
        soft[30:50, 40:60] = 100
        hard = np.full((90, 160, 3), 240, dtype=np.uint8)

        self.assertLess(main.frame_difference_score(before, soft), 0.18)
        self.assertGreater(main.frame_difference_score(before, hard), 0.18)


def _face_path(start, end, cx0, cx1, cy=200, size=40):
    boxes = {}
    span = max(1, end - start - 1)
    for n in range(start, end):
        t = (n - start) / span
        cx = cx0 + (cx1 - cx0) * t
        boxes[n] = [int(cx - size / 2), cy - size // 2, int(cx + size / 2), cy + size // 2]
    return boxes


class OverscanRenderTests(unittest.TestCase):
    def test_render_region_paints_only_the_source_intersection(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"numpy unavailable: {exc}")
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(160, dtype=np.uint8)[None, :]
        frame[:, :, 1] = 200
        out = main.render_region(frame, -20, 50, 160, 90, 50, 90, y=0, height=90)
        self.assertEqual(out.shape, (90, 50, 3))
        # The first 20 output columns are the overhang: black, not source.
        self.assertTrue(np.all(out[:, :20, 1] == 0))
        # Column 20 of the crop is source column 0.
        self.assertEqual(int(out[45, 20, 1]), 200)
        self.assertEqual(int(out[45, 20, 0]), 0)
        self.assertEqual(int(out[45, 21, 0]), 1)

    def test_full_bleed_crop_is_unchanged_when_it_fits(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"numpy unavailable: {exc}")
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(160, dtype=np.uint8)[None, :]
        frame[:, :, 1] = 200
        out = main.render_region(frame, 10, 50, 160, 90, 50, 90, y=0, height=90)
        self.assertEqual(int(out[0, 0, 0]), 10)
        self.assertEqual(int(out[0, 0, 1]), 200)
        self.assertTrue(np.all(out[:, :, 1] == 200))


class FollowCameraTests(unittest.TestCase):
    W, H, FPS = 1920, 1080, 30

    def _scene(self, start, end, cx0, cx1, boundary=None, speaking=None, cy=400):
        return {
            "start_frame": start,
            "end_frame": end,
            "strategy": "TRACK",
            "target_box": [cx0 - 20, cy - 20, cx0 + 20, cy + 20],
            "face_boxes": _face_path(start, end, cx0, cx1, cy=cy),
            "speaking_frames": speaking,
            "boundary_source": boundary,
        }

    def test_camera_follows_a_moving_face_inside_the_dead_zone_limit(self):
        # Face walks 400px. The camera must move with it, but not by more
        # than one crop-width per CAMERA_CROSS_SEC, and not in one frame.
        scene = self._scene(0, 90, 700, 1100)
        main.plan_follow_camera([scene], self.W, self.H, self.FPS)
        regions = [scene["follow_regions"][n] for n in range(90)]
        centers = [r[0] + r[2] / 2 for r in regions]
        self.assertAlmostEqual(centers[0], 700, delta=2)
        self.assertGreater(centers[-1], 1000)
        steps = [b - a for a, b in zip(centers, centers[1:])]
        crop_w = int(self.H * 9 / 16)
        max_step = crop_w / (main.CAMERA_CROSS_SEC * self.FPS)
        self.assertTrue(all(s <= max_step + 1.5 for s in steps))
        self.assertGreater(max(steps), 1)
        # It does not sit still and then jump.
        self.assertGreater(len({int(c) for c in centers}), 10)

    def test_dead_zone_ignores_detector_jitter(self):
        boxes = {}
        for n in range(30):
            cx = 960 + (3 if n % 2 else -3)
            boxes[n] = [cx - 20, 380, cx + 20, 420]
        scene = {
            "start_frame": 0, "end_frame": 30, "strategy": "TRACK",
            "target_box": [940, 380, 980, 420],
            "face_boxes": boxes, "speaking_frames": None,
        }
        main.plan_follow_camera([scene], self.W, self.H, self.FPS)
        origins = {scene["follow_regions"][n][0] for n in range(30)}
        self.assertEqual(len(origins), 1)

    def test_silence_holds_the_last_target(self):
        # Talking while the face sits at x=800, then silent while it walks off.
        speaking = list(range(0, 20))
        scene = self._scene(0, 80, 800, 1400, speaking=speaking)
        # The helper linearly moves the whole scene. Rebuild so the face is
        # still during speech and only walks once the scores go quiet.
        boxes = {}
        for n in range(80):
            cx = 800 if n < 20 else 800 + (n - 20) * 15
            boxes[n] = [cx - 20, 380, cx + 20, 420]
        scene["face_boxes"] = boxes
        main.plan_follow_camera([scene], self.W, self.H, self.FPS)
        held = {scene["follow_regions"][n][0] for n in range(20, 80)}
        self.assertEqual(held, {scene["follow_regions"][19][0]})
        self.assertAlmostEqual(
            scene["follow_regions"][19][0] + scene["follow_regions"][19][2] / 2,
            800, delta=2)

    def test_speaker_switch_glides_and_a_cut_snaps(self):
        a = self._scene(0, 30, 400, 400)
        b = self._scene(30, 90, 1500, 1500, boundary="speaker-turn")
        cut = self._scene(90, 120, 400, 400, boundary=None)
        scenes = [a, b, cut]
        main.plan_follow_camera(scenes, self.W, self.H, self.FPS)
        main.plan_pan_transitions(None, scenes, self.W, self.H, self.FPS, pan_duration=0.4)
        self.assertEqual(b["boundary_kind"], "follow")
        self.assertIsNone(b["transition"])
        crop_w = scenes[0]["follow_regions"][0][2]
        max_step = crop_w / (main.CAMERA_CROSS_SEC * self.FPS)
        before = a["follow_regions"][29]
        after = b["follow_regions"][30]
        self.assertLess(abs((after[0] + after[2] / 2) - (before[0] + before[2] / 2)), max_step + 2)
        # The glide arrives, it does not stay parked at the old face.
        arrived = b["follow_regions"][89]
        self.assertAlmostEqual(arrived[0] + arrived[2] / 2, 1500, delta=max_step + 2)
        # A real cut does not glide across the boundary.
        snapped = cut["follow_regions"][90]
        self.assertAlmostEqual(snapped[0] + snapped[2] / 2, 400, delta=2)
        self.assertEqual(main.summarize_pan_plan(scenes)["follow"], 1)
        self.assertEqual(main.summarize_pan_plan(scenes)["pan"], 0)

    def test_letterbox_boundary_still_zooms(self):
        wide = {"start_frame": 0, "end_frame": 20, "strategy": "LETTERBOX", "target_box": None}
        talk = self._scene(20, 50, 400, 400, boundary="speaker-turn")
        scenes = [wide, talk]
        main.plan_follow_camera(scenes, self.W, self.H, self.FPS)
        main.plan_pan_transitions(None, scenes, self.W, self.H, self.FPS,
                                  pan_duration=0.4, zoom_duration=0.5)
        self.assertEqual(talk["boundary_kind"], "zoom-in")
        self.assertEqual(talk["transition"]["kind"], "zoom-in")


class FfmpegCropCommandTests(unittest.TestCase):
    def test_pad_shifts_a_negative_crop_into_the_frame(self):
        regions = [(-8, 0, 100, 200), (10, -4, 100, 200)]
        pad_x, pad_y, pad_w, pad_h = main.source_pad(regions, 1920, 1080)
        self.assertEqual((pad_x, pad_y, pad_w, pad_h), (8, 4, 1928, 1084))
        script = main.crop_sendcmd(regions, 10, pad_x, pad_y)
        self.assertIn("0.000000 crop x 0, crop y 4, crop w 100, crop h 200;", script)
        self.assertIn("0.100000 crop x 18, crop y 0, crop w 100, crop h 200;", script)
        self.assertNotIn("crop x -", script)
        self.assertNotIn("crop y -", script)

    def test_encode_graph_changes_crop_size_in_its_own_segment(self):
        import tempfile
        regions = [(-8, 0, 100, 200), (10, -4, 80, 160)]
        with tempfile.TemporaryDirectory() as directory:
            graph = main.build_encode_graph(regions, 30, 1920, 1080, 608, 1080, directory)
        self.assertNotIn("rawvideo", graph)
        self.assertEqual(len(main.same_size_runs(regions)), 2)
        self.assertIn("interleave=nb_inputs=3", graph)
        self.assertIn("trim=end_frame=2", graph)
        self.assertIn("crop@run0=w=100:h=200:x=0:y=4", graph)
        self.assertIn("crop@run1=w=80:h=160:x=18:y=0", graph)

    def test_ffmpeg_applies_a_different_crop_on_each_frame(self):
        import os
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg is not installed")
        regions = [(0, 0, 16, 32), (16, 0, 16, 32)]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.mp4")
            output = os.path.join(directory, "out.mp4")
            cmd = os.path.join(directory, "crops.cmd")
            built = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=16x32:r=10:d=0.2",
                    "-f", "lavfi", "-i", "color=c=blue:s=16x32:r=10:d=0.2",
                    "-filter_complex", "hstack",
                    "-frames:v", "2",
                    "-pix_fmt", "yuv420p",
                    source,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr.decode()[-500:])
            code, stderr = main.run_ffmpeg_encode(
                source, output, regions, 10, 32, 32, 16, 32,
                "libx264", ["-preset", "ultrafast", "-crf", "18"], cmd,
            )
            self.assertEqual(code, 0, stderr[-500:])
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", output, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(raw.returncode, 0, raw.stderr.decode()[-500:])
            frame_bytes = 16 * 32 * 3
            self.assertGreaterEqual(len(raw.stdout), frame_bytes * 2)

            def channel_mean(buf, channel):
                return sum(buf[channel::3]) / (len(buf) / 3)

            first = raw.stdout[:frame_bytes]
            second = raw.stdout[frame_bytes:frame_bytes * 2]
            self.assertGreater(channel_mean(first, 0), channel_mean(first, 2) + 40)
            self.assertGreater(channel_mean(second, 2), channel_mean(second, 0) + 40)

    def test_ffmpeg_pans_after_a_crop_size_change(self):
        import os
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        regions = [(0, 0, 32, 32), (0, 0, 32, 32), (0, 0, 16, 32), (16, 0, 16, 32)]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.mp4")
            output = os.path.join(directory, "out.mp4")
            built = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=16x32:r=10:d=0.4",
                    "-f", "lavfi", "-i", "color=c=blue:s=16x32:r=10:d=0.4",
                    "-filter_complex", "hstack",
                    "-frames:v", "4",
                    "-pix_fmt", "yuv420p",
                    source,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr.decode()[-500:])
            code, stderr = main.run_ffmpeg_encode(
                source, output, regions, 10, 32, 32, 16, 32,
                "libx264", ["-preset", "ultrafast", "-crf", "18"],
                os.path.join(directory, "crops"),
            )
            self.assertEqual(code, 0, stderr[-500:])
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", output, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(raw.returncode, 0, raw.stderr.decode()[-500:])
            frame_bytes = 16 * 32 * 3
            self.assertGreaterEqual(len(raw.stdout), frame_bytes * 4)

            def channel_mean(buf, channel):
                return sum(buf[channel::3]) / (len(buf) / 3)

            left = raw.stdout[frame_bytes * 2:frame_bytes * 3]
            right = raw.stdout[frame_bytes * 3:frame_bytes * 4]
            self.assertGreater(channel_mean(left, 0), channel_mean(left, 2) + 40)
            self.assertGreater(channel_mean(right, 2), channel_mean(right, 0) + 40)

    def test_ffmpeg_zoom_changes_crop_height(self):
        import os
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        regions = [(0, 0, 32, 32), (0, 0, 32, 16)]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.mp4")
            output = os.path.join(directory, "out.mp4")
            built = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=32x16:r=10:d=0.2",
                    "-f", "lavfi", "-i", "color=c=blue:s=32x16:r=10:d=0.2",
                    "-filter_complex", "vstack",
                    "-frames:v", "2",
                    "-pix_fmt", "yuv420p",
                    source,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr.decode()[-500:])
            code, stderr = main.run_ffmpeg_encode(
                source, output, regions, 10, 32, 32, 16, 16,
                "libx264", ["-preset", "ultrafast", "-crf", "18"],
                os.path.join(directory, "crops"),
            )
            self.assertEqual(code, 0, stderr[-500:])
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", output, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(raw.returncode, 0, raw.stderr.decode()[-500:])
            frame_bytes = 16 * 16 * 3
            self.assertGreaterEqual(len(raw.stdout), frame_bytes * 2)

            def channel_mean(buf, channel):
                return sum(buf[channel::3]) / (len(buf) / 3)

            first_bottom = raw.stdout[:frame_bytes][16 * 12 * 3:]
            second_bottom = raw.stdout[frame_bytes:frame_bytes * 2][16 * 12 * 3:]
            self.assertGreater(channel_mean(first_bottom, 2), 80)
            self.assertLess(channel_mean(second_bottom, 2), 40)

    def test_ffmpeg_keeps_every_frame_when_each_crop_size_is_one_frame(self):
        import os
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        frame_count = 8
        regions = [(0, 0, 32, 16 + i) for i in range(frame_count)]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.mp4")
            output = os.path.join(directory, "out.mp4")
            built = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=1",
                    "-vf", "geq=r='N*30':g=0:b=0",
                    "-frames:v", str(frame_count),
                    "-pix_fmt", "yuv420p",
                    source,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr.decode()[-500:])
            code, stderr = main.run_ffmpeg_encode(
                source, output, regions, 10, 32, 32, 16, 16,
                "libx264", ["-preset", "ultrafast", "-crf", "0"],
                os.path.join(directory, "crops"),
            )
            self.assertEqual(code, 0, stderr[-500:])
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", output, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(raw.returncode, 0, raw.stderr.decode()[-500:])
            frame_bytes = 16 * 16 * 3
            self.assertEqual(len(raw.stdout) // frame_bytes, frame_count)

            def center_red(buf):
                x, y = 8, 8
                return buf[(y * 16 + x) * 3]

            for index in range(frame_count):
                frame = raw.stdout[index * frame_bytes:(index + 1) * frame_bytes]
                red = center_red(frame)
                self.assertGreater(red, index * 30 - 12, f"frame {index} red {red}")
                self.assertLess(red, index * 30 + 12, f"frame {index} red {red}")


if __name__ == "__main__":
    unittest.main()
