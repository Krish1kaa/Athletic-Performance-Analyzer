"""
Athletic Performance Analyzer

Analyzes athletic videos with MediaPipe Pose (Tasks API) to estimate performance metrics:
  - Vertical jump (airtime and normalized jump height)
  - Sprint time across the frame
  - Horizontal jump distance
  - Peak wrist velocity
  - Knee and hip flexion angles

Results can be saved per user to Cloud Firestore.

Setup:
    pip install -r requirements.txt
    # Download the pose model once (see README), e.g. into the project folder:
    #   pose_landmarker_lite.task

Usage:
    export FIREBASE_CREDENTIALS=path/to/serviceAccountKey.json   # optional
    export APA_POSE_MODEL=path/to/pose_landmarker_lite.task      # optional
    python athletic_performance_analyzer.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import firebase_admin
import mediapipe as mp
import numpy as np
from firebase_admin import credentials, firestore
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CREDENTIALS_ENV_VAR = "FIREBASE_CREDENTIALS"
DEFAULT_CREDENTIALS_PATH = "serviceAccountKey.json"
MODEL_ENV_VAR = "APA_POSE_MODEL"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
OUTPUT_DIR = Path("outputs")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".3gp"}
POSE_CONFIDENCE = 0.5
NUM_LANDMARKS = 33
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Landmark:
    """MediaPipe Pose landmark indices used by this script."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# Skeleton edges between landmarks, used for drawing.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


# ---------------------------------------------------------------------------
# Firebase
# ---------------------------------------------------------------------------

def initialize_firebase():
    """Return a Firestore client, or None if credentials are missing or invalid."""
    cred_path = Path(os.environ.get(CREDENTIALS_ENV_VAR, DEFAULT_CREDENTIALS_PATH))
    if not cred_path.is_file():
        print(f"Firebase credentials not found at '{cred_path}'.")
        print(f"Set the {CREDENTIALS_ENV_VAR} environment variable to your service account key path.")
        print("Get one from Firebase Console > Project Settings > Service Accounts > Generate New Private Key.")
        return None

    try:
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(credentials.Certificate(str(cred_path)))
        print("Firebase initialized.")
        return firestore.client()
    except Exception as exc:
        print(f"Error initializing Firebase: {exc}")
        return None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def _find_user_id(db, username: str):
    matches = db.collection("users").where("username", "==", username).limit(1).get()
    return matches[0].id if matches else None


def get_or_create_user(db):
    """Prompt for login or sign-up. Returns (user_id, username) or (None, None)."""
    print("\n" + "=" * 50)
    print("USER AUTHENTICATION")
    print("=" * 50)
    print("1. Log in with an existing username")
    print("2. Create a new user")
    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "1":
        username = input("Enter your username: ").strip()
        user_id = _find_user_id(db, username) if username else None
        if user_id:
            print(f"Welcome back, {username}!")
            return user_id, username
        print(f"Username '{username}' not found.")
        if input("Create this user? (y/n): ").strip().lower() == "y":
            return create_user(db, username)
        return None, None

    if choice == "2":
        return create_user(db, input("Enter a new username: ").strip())

    return None, None


def create_user(db, username: str):
    if not username:
        print("Username cannot be empty.")
        return None, None
    if _find_user_id(db, username):
        print(f"Username '{username}' already exists.")
        return None, None

    _, ref = db.collection("users").add(
        {"username": username, "created_at": firestore.SERVER_TIMESTAMP}
    )
    print(f"User '{username}' created.")
    return ref.id, username


# ---------------------------------------------------------------------------
# Firestore storage
# ---------------------------------------------------------------------------

def save_performance(db, user_id: str, data: dict) -> bool:
    record = {**data, "timestamp": int(time.time() * 1000)}
    try:
        (
            db.collection("users")
            .document(user_id)
            .collection("performance_records")
            .add(record)
        )
        print("Performance data saved.")
        return True
    except Exception as exc:
        print(f"Error saving to Firebase: {exc}")
        return False


def get_user_records(db, user_id: str, limit: int = 10) -> list[dict]:
    try:
        query = (
            db.collection("users")
            .document(user_id)
            .collection("performance_records")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        return [doc.to_dict() for doc in query.stream()]
    except Exception as exc:
        print(f"Error retrieving records: {exc}")
        return []


# ---------------------------------------------------------------------------
# Video input / output helpers
# ---------------------------------------------------------------------------

def prompt_for_video(label: str) -> Path | None:
    """Ask the user for a video path. Returns None if skipped."""
    print(f"\nProvide a video for: {label}")
    print("Enter a full path, drag the file into the terminal, or enter a filename")
    print("in the current directory. Leave blank to skip.")

    while True:
        raw = input("Video path: ").strip().strip('"').strip("'")
        if not raw:
            return None

        path = Path(raw).expanduser()
        if not path.is_file():
            print(f"File not found: {raw}")
        elif path.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"Unsupported format. Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        else:
            return path

        if input("Try again? (y/n): ").strip().lower() != "y":
            return None


def output_file(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def open_video(path: Path) -> None:
    """Open a video with the system's default player. Failures are non-fatal."""
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"Could not open video automatically ({exc}). Open it manually: {path}")


# ---------------------------------------------------------------------------
# Pose extraction (MediaPipe Tasks API)
# ---------------------------------------------------------------------------

def _model_path() -> Path:
    return Path(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_PATH))


def create_pose_landmarker(model_path: Path):
    """Create a video-mode PoseLandmarker. Raises RuntimeError if the model is missing."""
    if not model_path.is_file():
        raise RuntimeError(
            f"Pose model not found at '{model_path}'.\n"
            f"Download it from:\n  {MODEL_DOWNLOAD_URL}\n"
            f"then save it there, or set {MODEL_ENV_VAR} to its path."
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=POSE_CONFIDENCE,
        min_pose_presence_confidence=POSE_CONFIDENCE,
        min_tracking_confidence=POSE_CONFIDENCE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(video_path: Path):
    """
    Run pose estimation on every frame of a video.

    Returns (landmarks, fps, (width, height)) where landmarks is a list with one
    entry per frame: a (33, 2) float array of normalized x, y coordinates, or
    None if no person was detected in that frame.
    Raises RuntimeError if the video or model cannot be loaded.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    landmarks: list[np.ndarray | None] = []
    try:
        landmarker = create_pose_landmarker(_model_path())
    except Exception:
        cap.release()
        raise

    with landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                pts = np.array(
                    [[lm.x, lm.y] for lm in result.pose_landmarks[0]], dtype=float
                )
                landmarks.append(pts)
            else:
                landmarks.append(None)
            frame_idx += 1

    cap.release()
    return landmarks, fps, size


def draw_skeleton(frame: np.ndarray, pts: np.ndarray, size: tuple[int, int]) -> None:
    """Draw pose connections and joints onto a frame."""
    w, h = size
    px = (pts * [w, h]).astype(int)
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), (245, 117, 66), 2)
    for x, y in px:
        cv2.circle(frame, (int(x), int(y)), 3, (245, 66, 230), -1)


def write_annotated_video(video_path: Path, output_path: Path, landmarks, fps, size, overlay=None) -> Path:
    """
    Re-read the source video and write an annotated copy using landmarks
    already computed by extract_landmarks (pose runs only once per video).
    `overlay(frame, index, pts, size)` may draw extra graphics.
    """
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        for idx, pts in enumerate(landmarks):
            ok, frame = cap.read()
            if not ok:
                break
            if pts is not None:
                draw_skeleton(frame, pts, size)
            if overlay is not None:
                overlay(frame, idx, pts, size)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return output_path


def _point(pts: np.ndarray, part: int) -> np.ndarray:
    return pts[part]


def _midpoint(pts: np.ndarray, a: int, b: int) -> np.ndarray:
    return (pts[a] + pts[b]) / 2


def _to_pixels(pts: np.ndarray, part: int, size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    x, y = pts[part] * [w, h]
    return int(x), int(y)


def calculate_angle(a, b, c) -> float:
    """Interior angle in degrees at vertex b, formed by points a-b-c (2D)."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    cross = ba[0] * bc[1] - ba[1] * bc[0]
    # atan2 form is numerically stable near 0 and 180 degrees.
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def _print_report(title: str, lines: list[str]) -> None:
    print(f"\n--- {title} ---")
    for line in lines:
        print(line)
    print("-" * (len(title) + 8))


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analyze_jump(video_path: Path) -> dict | None:
    """
    Estimate airtime (seconds) and jump height. Height is normalized to frame
    height (0 to 1), not centimeters.
    """
    print("\nAnalyzing vertical jump...")
    landmarks, fps, size = extract_landmarks(video_path)

    ankle_y = np.array(
        [_midpoint(p, Landmark.LEFT_ANKLE, Landmark.RIGHT_ANKLE)[1] if p is not None else np.nan for p in landmarks]
    )
    hip_y = np.array(
        [_midpoint(p, Landmark.LEFT_HIP, Landmark.RIGHT_HIP)[1] if p is not None else np.nan for p in landmarks]
    )

    if len(ankle_y) == 0 or np.all(np.isnan(ankle_y)):
        print("No pose detected in the video.")
        return None

    # Baseline from the first 15 frames (standing still)
    baseline_window = ankle_y[:15]
    baseline = np.nanmean(baseline_window)
    threshold = max(np.nanstd(baseline_window) * 3, 0.01)

    takeoff = next(
        (i for i in range(15, len(ankle_y)) if ankle_y[i] < baseline - threshold), None
    )
    landing = None
    if takeoff is not None:
        landing = next(
            (i for i in range(takeoff + 1, len(ankle_y)) if abs(ankle_y[i] - baseline) < threshold / 2),
            None,
        )

    airtime = jump_height = None
    if takeoff is not None and landing is not None:
        airtime = (landing - takeoff) / fps
        crouch_hip = np.nanmax(hip_y[max(0, takeoff - 15):takeoff])
        peak_hip = np.nanmin(hip_y[takeoff:landing + 1])
        height = float(crouch_hip - peak_hip)
        if np.isfinite(height):
            jump_height = height

    def overlay(frame, idx, _pts, _size):
        if idx == takeoff:
            cv2.putText(frame, "TAKEOFF", (50, 50), FONT, 1, (0, 255, 0), 2)
        if idx == landing:
            cv2.putText(frame, "LANDING", (50, 100), FONT, 1, (0, 0, 255), 2)

    out = write_annotated_video(video_path, output_file("jump_height_output.mp4"), landmarks, fps, size, overlay)

    record: dict[str, float] = {}
    lines = []
    if airtime is not None:
        record["airtime"] = round(airtime, 3)
        lines.append(f"Airtime: {airtime:.3f} s")
    else:
        lines.append("Airtime: could not be determined")
    if jump_height is not None:
        record["jump_height"] = round(jump_height, 3)
        lines.append(f"Jump height (normalized): {jump_height:.3f}")
    else:
        lines.append("Jump height: could not be determined")

    _print_report("JUMP REPORT", lines)
    print(f"Annotated video saved to '{out}'")
    open_video(out)
    return record or None


def _first_crossing(values: np.ndarray, line: float, start_idx: int) -> int | None:
    """First index where the series moves from below `line` to at or above it."""
    for i in range(max(1, start_idx), len(values)):
        if values[i - 1] < line <= values[i]:
            return i
    return None


def analyze_sprint(video_path: Path) -> float | None:
    """
    Time (seconds) for the hips to travel from 15% to 85% of frame width.
    This equals a 20 m sprint only if the camera is set up so that span covers 20 m.
    """
    print("\nAnalyzing sprint time...")
    landmarks, fps, size = extract_landmarks(video_path)

    hip_x = np.array(
        [_midpoint(p, Landmark.LEFT_HIP, Landmark.RIGHT_HIP)[0] if p is not None else np.nan for p in landmarks]
    )
    if len(hip_x) == 0 or np.all(np.isnan(hip_x)):
        print("No pose detected in the video.")
        return None

    start = _first_crossing(hip_x, 0.15, 1)
    finish = _first_crossing(hip_x, 0.85, start) if start is not None else None

    sprint_time = (finish - start) / fps if start is not None and finish is not None else None

    if sprint_time is not None:
        _print_report("SPRINT REPORT", [f"Sprint time: {sprint_time:.3f} s"])
    else:
        _print_report("SPRINT REPORT", ["Could not detect crossing of both sprint lines."])

    out = write_annotated_video(video_path, output_file("sprint_time_output.mp4"), landmarks, fps, size)
    print(f"Annotated video saved to '{out}'")
    open_video(out)
    return round(sprint_time, 3) if sprint_time is not None else None


def analyze_horizontal_jump(video_path: Path) -> float | None:
    """Horizontal distance from the first to the last frames, normalized to frame width."""
    print("\nAnalyzing horizontal jump...")
    landmarks, fps, size = extract_landmarks(video_path)

    ankle_x = np.array(
        [_midpoint(p, Landmark.LEFT_ANKLE, Landmark.RIGHT_ANKLE)[0] if p is not None else np.nan for p in landmarks]
    )
    if len(ankle_x) == 0 or np.all(np.isnan(ankle_x)):
        print("No pose detected in the video.")
        return None

    start_x = np.nanmean(ankle_x[:10])
    end_x = np.nanmean(ankle_x[-10:])
    distance = float(abs(end_x - start_x)) if np.isfinite(start_x) and np.isfinite(end_x) else None

    if distance is not None:
        _print_report(
            "HORIZONTAL JUMP REPORT",
            [f"Distance (normalized): {distance:.3f}", "Note: value is proportional to frame width."],
        )
    else:
        _print_report("HORIZONTAL JUMP REPORT", ["Distance could not be determined."])

    out = write_annotated_video(video_path, output_file("horizontal_jump_output.mp4"), landmarks, fps, size)
    print(f"Annotated video saved to '{out}'")
    open_video(out)
    return round(distance, 3) if distance is not None else None


def analyze_peak_velocity(video_path: Path) -> float | None:
    """Peak speed of the right wrist in normalized frame units per second."""
    print("\nAnalyzing peak limb velocity...")
    landmarks, fps, size = extract_landmarks(video_path)

    wrist = np.array(
        [_point(p, Landmark.RIGHT_WRIST) if p is not None else [np.nan, np.nan] for p in landmarks],
        dtype=float,
    ).reshape(-1, 2)

    peak = None
    if len(wrist) > 1:
        speeds = np.linalg.norm(np.diff(wrist, axis=0), axis=1) * fps
        speeds = speeds[np.isfinite(speeds)]
        if speeds.size:
            peak = float(speeds.max())

    if peak is not None:
        _print_report("PEAK VELOCITY REPORT", [f"Peak wrist velocity (normalized): {peak:.3f} units/s"])
    else:
        _print_report("PEAK VELOCITY REPORT", ["Peak velocity could not be determined."])

    out = write_annotated_video(video_path, output_file("peak_velocity_output.mp4"), landmarks, fps, size)
    print(f"Annotated video saved to '{out}'")
    open_video(out)
    return round(peak, 3) if peak is not None else None


def analyze_flexion(video_path: Path) -> dict | None:
    """Minimum (most flexed) left knee and hip angles across the video, in degrees."""
    print("\nAnalyzing knee and hip flexion...")
    landmarks, fps, size = extract_landmarks(video_path)

    angles: list[tuple[float, float] | None] = []
    for pts in landmarks:
        if pts is None:
            angles.append(None)
            continue
        shoulder = _point(pts, Landmark.LEFT_SHOULDER)
        hip = _point(pts, Landmark.LEFT_HIP)
        knee = _point(pts, Landmark.LEFT_KNEE)
        ankle = _point(pts, Landmark.LEFT_ANKLE)
        angles.append((calculate_angle(hip, knee, ankle), calculate_angle(shoulder, hip, knee)))

    valid = [a for a in angles if a is not None]
    if not valid:
        print("No pose detected in the video.")
        return None

    min_knee = min(k for k, _ in valid)
    min_hip = min(h for _, h in valid)

    def overlay(frame, idx, pts, frame_size):
        if angles[idx] is None or pts is None:
            return
        knee_deg, hip_deg = angles[idx]
        cv2.putText(frame, f"Knee: {int(knee_deg)}", _to_pixels(pts, Landmark.LEFT_KNEE, frame_size),
                    FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Hip: {int(hip_deg)}", _to_pixels(pts, Landmark.LEFT_HIP, frame_size),
                    FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)

    out = write_annotated_video(video_path, output_file("flexion_angles_output.mp4"), landmarks, fps, size, overlay)

    record = {
        "min_knee_flexion_angle": round(min_knee, 2),
        "min_hip_flexion_angle": round(min_hip, 2),
    }
    _print_report(
        "FLEXION ANGLE REPORT",
        [
            f"Minimum knee angle: {min_knee:.2f} degrees",
            f"Minimum hip angle: {min_hip:.2f} degrees",
        ],
    )
    print(f"Annotated video saved to '{out}'")
    open_video(out)
    return record


# ---------------------------------------------------------------------------
# Menu flows
# ---------------------------------------------------------------------------

def _identity(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class Analysis:
    key: str                                     # Firestore field name
    label: str                                   # Menu label
    instructions: str                            # Shown when asking for a video
    run: Callable[[Path], Any]                   # Analyzer function
    to_record: Callable[[Any], Any] = _identity  # Converts result to stored form


ANALYSES = [
    Analysis("vertical_jump", "Vertical jump (height / airtime)",
             "a vertical jump", analyze_jump),
    Analysis("sprint_time_20m", "Sprint time",
             "a sprint running left to right across the frame", analyze_sprint),
    Analysis("horizontal_jump", "Horizontal jump distance",
             "a standing horizontal jump", analyze_horizontal_jump,
             to_record=lambda d: {"distance": d}),
    Analysis("peak_limb_velocity", "Peak wrist velocity",
             "a throwing or striking motion", analyze_peak_velocity),
    Analysis("knee_hip_flexion", "Knee & hip flexion angles",
             "a squat or similar movement", analyze_flexion),
]


def run_analysis(spec: Analysis) -> Any | None:
    """Prompt for a video, run the analysis, and return the record to store."""
    video = prompt_for_video(f"{spec.label} ({spec.instructions})")
    if video is None:
        return None
    try:
        result = spec.run(video)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return None
    return spec.to_record(result) if result is not None else None


def comprehensive_analysis(db, user_id) -> None:
    print("\n" + "=" * 50)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    print("You will be asked for a video for each metric. Leave a prompt blank to skip it.")

    data = {}
    for number, spec in enumerate(ANALYSES, 1):
        print(f"\n[{number}/{len(ANALYSES)}] {spec.label}")
        record = run_analysis(spec)
        if record is not None:
            data[spec.key] = record

    if not data:
        print("\nNo performance data was collected.")
        return

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for key, value in data.items():
        print(f"  {key}: {value}")

    if db and user_id:
        if input("\nSave to Firebase? (y/n): ").strip().lower() == "y":
            save_performance(db, user_id, data)
    else:
        print("\nNot logged in to Firebase; results were not saved.")


def individual_analysis(db, user_id) -> None:
    print("\n" + "=" * 50)
    print("SELECT A METRIC")
    print("=" * 50)
    for number, spec in enumerate(ANALYSES, 1):
        print(f"{number}. {spec.label}")
    print("0. Back")

    choice = input(f"\nEnter your choice (0-{len(ANALYSES)}): ").strip()
    if not choice.isdigit() or not 0 <= int(choice) <= len(ANALYSES):
        print("Invalid choice.")
        return
    if choice == "0":
        return

    spec = ANALYSES[int(choice) - 1]
    record = run_analysis(spec)
    if record is None:
        return

    print(f"\n{spec.label}: {record}")
    if db and user_id:
        if input("Save to Firebase? (y/n): ").strip().lower() == "y":
            save_performance(db, user_id, {spec.key: record})


def view_records(db, user_id, username: str) -> None:
    print("\n" + "=" * 50)
    print(f"PERFORMANCE HISTORY for {username}")
    print("=" * 50)

    records = get_user_records(db, user_id)
    if not records:
        print("No records found.")
        return

    for i, record in enumerate(records, 1):
        timestamp = record.get("timestamp", 0)
        when = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nRecord #{i} - {when}")
        print("-" * 30)
        for spec in ANALYSES:
            if spec.key not in record:
                continue
            value = record[spec.key]
            if isinstance(value, dict):
                details = ", ".join(f"{k}={v}" for k, v in value.items())
                print(f"  {spec.label}: {details}")
            else:
                print(f"  {spec.label}: {value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("ATHLETIC PERFORMANCE ANALYZER")
    print("=" * 60)

    db = initialize_firebase()
    user_id = username = None

    if db is None:
        if input("\nContinue without saving to Firebase? (y/n): ").strip().lower() != "y":
            return
    else:
        user_id, username = get_or_create_user(db)
        if user_id is None:
            print("No user logged in; results will not be saved.")

    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        if username:
            print(f"Logged in as: {username}")
            print("-" * 50)
        print("1. Comprehensive analysis (all metrics)")
        print("2. Individual metric analysis")
        print("3. View performance history")
        print("4. Switch user")
        print("5. Exit")
        print("=" * 50)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            comprehensive_analysis(db, user_id)
        elif choice == "2":
            individual_analysis(db, user_id)
        elif choice == "3":
            if db and user_id:
                view_records(db, user_id, username)
            else:
                print("Log in to view performance history.")
        elif choice == "4":
            if db:
                new_id, new_name = get_or_create_user(db)
                if new_id:
                    user_id, username = new_id, new_name
            else:
                print("Firebase is not connected.")
        elif choice == "5":
            print("\nThanks for using the Athletic Performance Analyzer.")
            break
        else:
            print("Invalid choice. Enter a number from 1 to 5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
