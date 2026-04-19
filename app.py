from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import sys
import subprocess
from flask import Flask, Response, jsonify, send_from_directory, request


BASE_DIR = Path(__file__).resolve().parent
TRACKING_FILE = BASE_DIR / "tracking_stream.json"
POLL_INTERVAL = 1 / 30
HISTORY_SIZE = 240

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


class TrackingFeed:
    def __init__(self, source: Path) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._frame: dict[str, Any] = self._demo_frame()
        self._history: deque[dict[str, Any]] = deque(maxlen=HISTORY_SIZE)
        self._history.append(self._frame)
        self._last_signature: tuple[float, float] | None = None

    def _demo_frame(self) -> dict[str, Any]:
        t = time.time()
        phase = t * 1.4
        x = 0.5 + math.sin(phase) * 0.24
        y = 0.5 + math.cos(phase * 0.85) * 0.2
        z = 0.5 + math.sin(phase * 1.2) * 0.18
        tilt = 0.5 + math.cos(phase * 0.55) * 0.22
        roll = 0.5 + math.sin(phase * 0.7) * 0.2
        joints = {
            "shoulder_pan": (x - 0.5) * 90,
            "shoulder_lift": (0.5 - y) * 60,
            "elbow_flex": (z - 0.5) * 60,
            "wrist_flex": 90 + (tilt - 0.5) * 55,
            "wrist_roll": 90 + (roll - 0.5) * 40,
        }
        return self._normalize(
            {
                "timestamp": t,
                "tracking_active": False,
                "gesture": "DEMO",
                "hand_data": {"x": x, "y": y, "z": z, "tilt": tilt, "roll": roll},
                "joint_positions": joints,
                "gripper_value": 50.0 + math.sin(phase * 2.2) * 35,
                "bounding_box": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
            }
        )

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        hand = payload.get("hand_data") or {}
        joints = payload.get("joint_positions") or {}
        ts = float(payload.get("timestamp") or time.time())
        x = float(hand.get("x", 0.5))
        y = float(hand.get("y", 0.5))
        z = float(hand.get("z", 0.5))
        tilt = float(hand.get("tilt", 0.5))
        roll = float(hand.get("roll", 0.5))
        gesture = str(payload.get("gesture") or "UNKNOWN")
        gripper = payload.get("gripper_value")
        gripper_value = None if gripper is None else float(gripper)

        motion_energy = min(
            1.0,
            abs(x - 0.5) * 0.85 + abs(y - 0.5) * 0.8 + abs(z - 0.5) * 1.15,
        )
        confidence = 0.92 if payload.get("tracking_active") else 0.38
        depth_cm = round(20 + z * 40, 1)

        return {
            "timestamp": ts,
            "tracking_active": bool(payload.get("tracking_active")),
            "gesture": gesture,
            "bounding_box": payload.get("bounding_box") or {},
            "hand_data": {
                "x": x,
                "y": y,
                "z": z,
                "tilt": tilt,
                "roll": roll,
            },
            "joint_positions": {
                "shoulder_pan": float(joints.get("shoulder_pan", 0.0)),
                "shoulder_lift": float(joints.get("shoulder_lift", 0.0)),
                "elbow_flex": float(joints.get("elbow_flex", 0.0)),
                "wrist_flex": float(joints.get("wrist_flex", 90.0)),
                "wrist_roll": float(joints.get("wrist_roll", 90.0)),
            },
            "gripper_value": gripper_value,
            "telemetry": {
                "confidence": confidence,
                "motion_energy": motion_energy,
                "depth_cm": depth_cm,
            },
            "t": ts,
            "px": x,
            "py": y,
            "fx": min(1.0, max(0.0, x + (roll - 0.5) * 0.18)),
            "fy": min(1.0, max(0.0, y - (tilt - 0.5) * 0.18)),
        }

    def refresh(self) -> dict[str, Any]:
        try:
            stat = self.source.stat()
            signature = (stat.st_mtime, stat.st_size)
            if signature == self._last_signature:
                return self.current_frame()

            payload = json.loads(self.source.read_text(encoding="utf-8"))
            frame = self._normalize(payload)

            with self._lock:
                self._last_signature = signature
                self._frame = frame
                self._history.append(frame)
                return frame
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            frame = self._demo_frame()
            with self._lock:
                self._frame = frame
                self._history.append(frame)
                return frame

    def current_frame(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._frame)

    def history(self, limit: int = 90) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)[-limit:]


feed = TrackingFeed(TRACKING_FILE)


def watcher() -> None:
    while True:
        feed.refresh()
        time.sleep(POLL_INTERVAL)


threading.Thread(target=watcher, daemon=True).start()


@app.route("/")
def index() -> Response:
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/dashboard")
def dashboard() -> Response:
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.get("/api/frame")
def api_frame() -> Response:
    return jsonify(feed.current_frame())


@app.get("/api/history")
def api_history() -> Response:
    return jsonify(feed.history())


@app.get("/api/stream")
def api_stream() -> Response:
    def generate() -> Any:
        last_ts = None
        while True:
            frame = feed.current_frame()
            ts = frame["timestamp"]
            if ts != last_ts:
                last_ts = ts
                yield f"data: {json.dumps(frame)}\n\n"
            time.sleep(POLL_INTERVAL)

    return Response(generate(), mimetype="text/event-stream")

# --- Process Control ---
arm_process = None

@app.post("/api/start_arm")
def start_arm() -> Response:
    global arm_process
    if arm_process is None or arm_process.poll() is not None:
        cmd = [sys.executable, "main.py", "--port", "COM10", "--robot-id", "gesture_follower"]
        arm_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=str(BASE_DIR))
        try:
            # Send a newline to quickly bypass the lerobot calibration prompt
            arm_process.stdin.write(b'\n')
            arm_process.stdin.flush()
        except Exception:
            pass
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.post("/api/stop_arm")
def stop_arm() -> Response:
    global arm_process
    if arm_process and arm_process.poll() is None:
        arm_process.terminate()
        arm_process = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
