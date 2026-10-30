import contextlib, io, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

import cv2
from firebase_admin import firestore
import numpy as np

import athletic_performance_analyzer as apa

FPS = 30.0
SIZE = (320, 240)


def make_video(path: Path, frames: int = 60) -> Path:
    """Write a real, readable mp4 with a moving rectangle."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(frames):
        img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        cv2.rectangle(img, (10 + i * 2, 100), (40 + i * 2, 200), (0, 255, 0), -1)
        w.write(img)
    w.release()
    return path


def pose(points: dict[int, tuple[float, float]] | None = None) -> np.ndarray:
    """A (33,2) landmark array; unspecified landmarks sit at (0.5, 0.5)."""
    pts = np.full((33, 2), 0.5)
    for k, v in (points or {}).items():
        pts[k] = v
    return pts


def standing_pose(hip_x=0.5, hip_y=0.5, ankle_y=0.9, wrist=(0.5, 0.5), knee=None, ankle_x=None):
    ax = hip_x if ankle_x is None else ankle_x
    p = {
        apa.Landmark.LEFT_HIP: (hip_x, hip_y), apa.Landmark.RIGHT_HIP: (hip_x, hip_y),
        apa.Landmark.LEFT_ANKLE: (ax, ankle_y), apa.Landmark.RIGHT_ANKLE: (ax, ankle_y),
        apa.Landmark.RIGHT_WRIST: wrist,
        apa.Landmark.LEFT_SHOULDER: (hip_x, hip_y - 0.3),
        apa.Landmark.LEFT_KNEE: knee if knee else (hip_x, hip_y + 0.2),
    }
    return pose(p)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp)  # outputs/ is created relative to cwd
        self.video = make_video(Path(self.tmp) / "src.mp4")
        self.patch_open = mock.patch.object(apa, "open_video", lambda p: None)
        self.patch_open.start()

    def tearDown(self):
        self.patch_open.stop()
        os.chdir(self.old_cwd)

    def fake_landmarks(self, frames, fps=FPS):
        return mock.patch.object(apa, "extract_landmarks", lambda _p: (frames, fps, SIZE))

    def quiet(self, fn, *args, inputs=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch("builtins.input", side_effect=inputs or []):
            result = fn(*args)
        return result, buf.getvalue()


class TestMath(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(apa.calculate_angle((1, 0), (0, 0), (0, 1)), 90.0, places=5)

    def test_straight_line(self):
        self.assertAlmostEqual(apa.calculate_angle((-1, 0), (0, 0), (1, 0)), 180.0, places=4)

    def test_folded(self):
        self.assertAlmostEqual(apa.calculate_angle((1, 0), (0, 0), (1, 0)), 0.0, places=4)

    def test_first_crossing(self):
        v = np.array([0.0, 0.1, 0.2, 0.6, 0.9])
        self.assertEqual(apa._first_crossing(v, 0.5, 1), 3)
        self.assertIsNone(apa._first_crossing(v, 0.95, 1))


class TestPoseErrors(Base):
    def test_missing_model_gives_clear_error(self):
        with mock.patch.dict(os.environ, {"APA_POSE_MODEL": str(Path(self.tmp) / "nope.task")}):
            with self.assertRaises(RuntimeError) as ctx:
                apa.extract_landmarks(self.video)
        self.assertIn("Pose model not found", str(ctx.exception))
        self.assertIn("storage.googleapis.com", str(ctx.exception))

    def test_unreadable_video(self):
        bad = Path(self.tmp) / "broken.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(RuntimeError):
            apa.extract_landmarks(bad)


class TestAnalyses(Base):
    def test_jump_detects_airtime_and_height(self):
        frames = []
        for i in range(60):
            ankle = 0.7 if 25 <= i < 35 else 0.9   # rises for 10 frames
            hip = ankle - 0.4
            frames.append(standing_pose(hip_y=hip, ankle_y=ankle))
        with self.fake_landmarks(frames):
            rec, out = self.quiet(apa.analyze_jump, self.video)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec["airtime"], 10 / FPS, delta=0.05)
        self.assertGreater(rec["jump_height"], 0.1)
        self.assertTrue((Path(self.tmp) / "outputs" / "jump_height_output.mp4").exists())

    def test_sprint_time_between_lines(self):
        frames = [standing_pose(hip_x=i / 59) for i in range(60)]
        with self.fake_landmarks(frames):
            t, _ = self.quiet(apa.analyze_sprint, self.video)
        # 0.15 -> 0.85 over 59 frames of x increments
        expected = (0.7 * 59) / FPS
        self.assertAlmostEqual(t, expected, delta=2 / FPS)

    def test_horizontal_jump(self):
        frames = [standing_pose(ankle_x=0.2 + 0.6 * i / 59) for i in range(60)]
        with self.fake_landmarks(frames):
            d, _ = self.quiet(apa.analyze_horizontal_jump, self.video)
        self.assertAlmostEqual(d, 0.5, delta=0.03)

    def test_peak_velocity(self):
        frames = [standing_pose(wrist=(0.1 + 0.01 * i, 0.5)) for i in range(20)]
        frames[10] = standing_pose(wrist=(0.5, 0.5))  # fast jump
        with self.fake_landmarks(frames):
            v, _ = self.quiet(apa.analyze_peak_velocity, self.video)
        self.assertGreater(v, 0.01 * FPS)

    def test_flexion_returns_dict_with_knee_angle(self):
        # Knee bends to 90 degrees in the middle of the clip.
        frames = []
        for i in range(30):
            if i == 15:
                knee = (0.6, 0.7)
            else:
                knee = (0.5, 0.7)
            frames.append(standing_pose(hip_x=0.5, hip_y=0.5, knee=knee))
        with self.fake_landmarks(frames):
            rec, _ = self.quiet(apa.analyze_flexion, self.video)
        self.assertIsNotNone(rec, "flexion analysis must return results (was a bug)")
        self.assertIn("min_knee_flexion_angle", rec)
        self.assertLess(rec["min_knee_flexion_angle"], 180)

    def test_no_pose_returns_none(self):
        frames = [None] * 30
        with self.fake_landmarks(frames):
            for fn in (apa.analyze_jump, apa.analyze_sprint, apa.analyze_horizontal_jump,
                       apa.analyze_peak_velocity, apa.analyze_flexion):
                result, out = self.quiet(fn, self.video)
                self.assertIsNone(result, fn.__name__)
                self.assertIn("no pose", out.lower(), fn.__name__)

    def test_real_video_path_with_no_model_reports_error(self):
        # Full path: real video decode, then clean error from missing model.
        with mock.patch.dict(os.environ, {"APA_POSE_MODEL": str(Path(self.tmp) / "none.task")}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), mock.patch("builtins.input", side_effect=[str(self.video), "n"]):
                rec = apa.run_analysis(apa.ANALYSES[0])
        self.assertIsNone(rec)
        self.assertIn("Pose model not found", buf.getvalue())


class FakeSnap:
    def __init__(self, id_, data): self.id, self._d = id_, data
    def to_dict(self): return dict(self._d)


class FakeQuery:
    def __init__(self, store, path, filters=None, order=None, limit=None):
        self.store, self.path, self.filters, self.order, self.lim = store, path, filters or [], order, limit
    def where(self, field, op, value):
        return FakeQuery(self.store, self.path, self.filters + [(field, value)], self.order, self.lim)
    def order_by(self, field, direction=None):
        return FakeQuery(self.store, self.path, self.filters, (field, direction), self.lim)
    def limit(self, n):
        return FakeQuery(self.store, self.path, self.filters, self.order, n)
    def _docs(self):
        docs = [(k, v) for k, v in self.store.items() if k.startswith(self.path + "/") and k.count("/") == self.path.count("/") + 1]
        docs = [FakeSnap(k.split("/")[-1], v) for k, v in docs]
        for f, val in self.filters:
            docs = [d for d in docs if d._d.get(f) == val]
        if self.order:
            field, direction = self.order
            docs.sort(key=lambda d: d._d.get(field, 0), reverse=(direction == firestore.Query.DESCENDING))
        if self.lim is not None:
            docs = docs[: self.lim]
        return docs
    def get(self): return self._docs()
    def stream(self): return iter(self._docs())


class FakeDoc:
    def __init__(self, store, path): self.store, self.path = store, path
    def collection(self, name): return FakeCol(self.store, f"{self.path}/{name}")


class FakeCol(FakeQuery):
    def __init__(self, store, path):
        super().__init__(store, path)
        self._next = 0
    def document(self, doc_id): return FakeDoc(self.store, f"{self.path}/{doc_id}")
    def add(self, data):
        self._next += 1
        doc_id = f"auto{len(self.store)}"
        self.store[f"{self.path}/{doc_id}"] = dict(data)
        ref = FakeDoc(self.store, f"{self.path}/{doc_id}")
        ref.id = doc_id
        return (None, ref)


class FakeDB:
    def __init__(self): self.store = {}
    def collection(self, name): return FakeCol(self.store, name)


class TestFirestoreFlows(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        apa.firestore.SERVER_TIMESTAMP = "TS"  # placeholder sentinel for the fake

    def test_create_then_login(self):
        with mock.patch("builtins.input", side_effect=["2", "kris"]), contextlib.redirect_stdout(io.StringIO()):
            uid, name = apa.get_or_create_user(self.db)
        self.assertIsNotNone(uid); self.assertEqual(name, "kris")
        with mock.patch("builtins.input", side_effect=["1", "kris"]), contextlib.redirect_stdout(io.StringIO()):
            uid2, name2 = apa.get_or_create_user(self.db)
        self.assertEqual((uid, name), (uid2, name2))

    def test_duplicate_username_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            apa.create_user(self.db, "kris")
            uid, _ = apa.create_user(self.db, "kris")
        self.assertIsNone(uid)

    def test_empty_username_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(apa.create_user(self.db, ""), (None, None))

    def test_save_and_history_ordered_newest_first(self):
        with contextlib.redirect_stdout(io.StringIO()):
            uid, _ = apa.create_user(self.db, "kris")
            with mock.patch.object(apa.time, "time", return_value=1_000):
                apa.save_performance(self.db, uid, {"sprint_time_20m": 3.1})
            with mock.patch.object(apa.time, "time", return_value=2_000):
                apa.save_performance(self.db, uid, {"horizontal_jump": {"distance": 0.4}})
            records = apa.get_user_records(self.db, uid)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["timestamp"], 2_000_000)
        self.assertEqual(records[1]["sprint_time_20m"], 3.1)

    def test_view_records_output(self):
        with contextlib.redirect_stdout(io.StringIO()):
            uid, _ = apa.create_user(self.db, "kris")
            apa.save_performance(self.db, uid, {"knee_hip_flexion": {"min_knee_flexion_angle": 80.5, "min_hip_flexion_angle": 60.1}})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            apa.view_records(self.db, uid, "kris")
        self.assertIn("min_knee_flexion_angle=80.5", buf.getvalue())

    def test_no_credentials_returns_none(self):
        with mock.patch.dict(os.environ, {"FIREBASE_CREDENTIALS": "/nonexistent/key.json"}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(apa.initialize_firebase())


class TestCli(unittest.TestCase):
    def test_offline_menu_flow_runs_and_exits(self):
        env = {**os.environ, "FIREBASE_CREDENTIALS": "/nonexistent/key.json"}
        # continue offline -> menu 2 (individual) -> 0 back -> menu 5 exit
        proc = subprocess.run(
            [sys.executable, str(Path(apa.__file__).resolve())],
            input="y\n2\n0\n5\n", text=True, capture_output=True, env=env, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Thanks for using", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
