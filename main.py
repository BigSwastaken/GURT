import cv2
import mediapipe as mp
import math
import time
import argparse
import numpy as np

# ============================================================
# Gesture Detection Helpers
# ============================================================

def finger_is_extended(hand_landmarks, finger_tip_id, finger_pip_id):
    """Check if a single finger is extended by comparing tip vs PIP y-coords."""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y


def thumb_is_extended(hand_landmarks):
    """Check if the thumb is extended."""
    tip = hand_landmarks.landmark[4]
    ip_joint = hand_landmarks.landmark[3]
    mcp = hand_landmarks.landmark[2]
    thumb_len = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
    thumb_base = math.hypot(ip_joint.x - mcp.x, ip_joint.y - mcp.y)
    return thumb_len > thumb_base * 1.2


def detect_gesture(hand_landmarks):
    """Classify hand gesture: 'OPEN', 'FIST', or 'OTHER'."""
    fingers_extended = 0
    if finger_is_extended(hand_landmarks, 8, 6):   fingers_extended += 1  # Index
    if finger_is_extended(hand_landmarks, 12, 10):  fingers_extended += 1  # Middle
    if finger_is_extended(hand_landmarks, 16, 14):  fingers_extended += 1  # Ring
    if finger_is_extended(hand_landmarks, 20, 18):  fingers_extended += 1  # Pinky
    if thumb_is_extended(hand_landmarks):            fingers_extended += 1  # Thumb

    if fingers_extended >= 4:
        return "OPEN", fingers_extended
    elif fingers_extended <= 1:
        return "FIST", fingers_extended
    else:
        return "OTHER", fingers_extended


# ============================================================
# Hand Position Tracker (X, Y, Z)
# ============================================================

class HandPositionTracker:
    """Extracts smoothed X, Y, Z hand position from MediaPipe landmarks.
    
    X = palm center horizontal position (0.0 = left, 1.0 = right)
    Y = palm center vertical position (0.0 = top, 1.0 = bottom)
    Z = estimated depth using palm apparent size (0.0 = far, 1.0 = close)
    
    Z uses PALM-ONLY landmarks (wrist, MCPs) which are stable regardless
    of whether the hand is open or in a fist.
    """

    def __init__(self, smoothing=0.4):
        """
        Args:
            smoothing: Exponential moving average factor (0 = no smoothing, 1 = max smoothing).
                       Higher = smoother but laggier.
        """
        self.smoothing = smoothing
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_z = None
        self.smooth_tilt = None
        self.smooth_roll = None
        # Reference palm size for Z estimation (calibrated on first detection)
        self.ref_palm_size = None

    def _measure_palm_size(self, hand_landmarks):
        """Measure palm size using landmarks that DON'T change with finger curl."""
        wrist = hand_landmarks.landmark[0]
        mid_mcp = hand_landmarks.landmark[9]
        idx_mcp = hand_landmarks.landmark[5]
        pinky_mcp = hand_landmarks.landmark[17]

        palm_length = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y)
        palm_width = math.hypot(idx_mcp.x - pinky_mcp.x, idx_mcp.y - pinky_mcp.y)
        return (palm_length + palm_width) / 2.0

    def _measure_tilt(self, hand_landmarks):
        """Measure hand tilt (pitch) for wrist_flex.

        Uses the angle of the wrist-to-MCP9 line relative to vertical.
        Returns normalized [0.0, 1.0]: 0.0 = hand pointing up, 1.0 = hand pointing down.
        """
        wrist = hand_landmarks.landmark[0]
        mid_mcp = hand_landmarks.landmark[9]

        # Angle from vertical: atan2(dx, -dy) so straight up = 0
        angle = math.atan2(mid_mcp.x - wrist.x, -(mid_mcp.y - wrist.y))
        # angle is in [-pi, pi], typical hand range is about [-1.2, 1.2] radians
        # Normalize to [0, 1]
        normalized = (angle + 1.2) / 2.4
        return max(0.0, min(1.0, normalized))

    def _measure_roll(self, hand_landmarks):
        """Measure hand roll (rotation) for wrist_roll.

        Uses the angle of the line from index MCP (5) to pinky MCP (17).
        Returns normalized [0.0, 1.0]: 0.5 = palm flat, 0.0/1.0 = rotated.
        """
        idx_mcp = hand_landmarks.landmark[5]
        pinky_mcp = hand_landmarks.landmark[17]

        # Angle of the palm width line relative to horizontal
        angle = math.atan2(pinky_mcp.y - idx_mcp.y, pinky_mcp.x - idx_mcp.x)
        # angle is in [-pi, pi], typical range is about [-1.0, 1.0] radians
        # Normalize to [0, 1]
        normalized = (angle + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    def update(self, hand_landmarks):
        """Extract and smooth X, Y, Z, tilt, roll from hand landmarks.
        
        Returns:
            (x, y, z, tilt, roll) tuple, all normalized to [0.0, 1.0]
        """
        # --- X and Y: Use landmark 9 (base of middle finger = palm center) ---
        palm = hand_landmarks.landmark[9]
        raw_x = palm.x
        raw_y = palm.y

        # --- Z: Estimate depth from apparent PALM size ---
        palm_size = self._measure_palm_size(hand_landmarks)
        if self.ref_palm_size is None:
            self.ref_palm_size = palm_size

        size_ratio = palm_size / self.ref_palm_size
        size_ratio = max(0.3, min(2.0, size_ratio))
        raw_z = (size_ratio - 0.3) / (2.0 - 0.3)

        # --- Tilt and Roll ---
        raw_tilt = self._measure_tilt(hand_landmarks)
        raw_roll = self._measure_roll(hand_landmarks)

        # --- Exponential moving average smoothing ---
        if self.smooth_x is None:
            self.smooth_x = raw_x
            self.smooth_y = raw_y
            self.smooth_z = raw_z
            self.smooth_tilt = raw_tilt
            self.smooth_roll = raw_roll
        else:
            a = self.smoothing
            self.smooth_x = a * self.smooth_x + (1 - a) * raw_x
            self.smooth_y = a * self.smooth_y + (1 - a) * raw_y
            self.smooth_z = a * self.smooth_z + (1 - a) * raw_z
            self.smooth_tilt = a * self.smooth_tilt + (1 - a) * raw_tilt
            self.smooth_roll = a * self.smooth_roll + (1 - a) * raw_roll

        return self.smooth_x, self.smooth_y, self.smooth_z, self.smooth_tilt, self.smooth_roll

    def reset(self):
        """Reset tracking state (call when hand is lost)."""
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_z = None
        self.smooth_tilt = None
        self.smooth_roll = None
        self.ref_palm_size = None


# ============================================================
# Arm Controller
# ============================================================

class ArmController:
    """Controls the SO-101 follower arm via the LeRobot API.
    
    Maps hand X, Y, Z → arm joint angles, and gesture → gripper.
    
    SO-101 motors (in degrees mode):
        shoulder_pan  (ID 1) - Base rotation (left/right)
        shoulder_lift (ID 2) - Shoulder up/down
        elbow_flex    (ID 3) - Elbow bend
        wrist_flex    (ID 4) - Wrist up/down
        wrist_roll    (ID 5) - Wrist rotation
        gripper       (ID 6) - Gripper open/close (0-100)
    """

    # Default joint angle ranges (degrees) — conservative to avoid mechanical damage
    # Format: (min_degrees, max_degrees, center_degrees)
    DEFAULT_RANGES = {
        "shoulder_pan":  (-45.0, 45.0, 0.0),    # Hand X -> rotate base
        "shoulder_lift": (-30.0, 30.0, 0.0),     # Hand Y -> raise/lower
        "elbow_flex":    (-30.0, 30.0, 0.0),     # Hand Z -> extend/retract
        "wrist_flex":    (-30.0, 30.0, 0.0),     # Hand tilt -> wrist up/down
        "wrist_roll":    (-45.0, 45.0, 0.0),     # Hand roll -> wrist rotation
    }

    def __init__(self, port=None, robot_id="gesture_follower"):
        self.port = port
        self.robot_id = robot_id
        self.robot = None
        self.connected = False
        self.last_gripper_cmd = None
        self.joint_ranges = dict(self.DEFAULT_RANGES)

    def connect(self):
        """Connect to the SO-101 follower arm."""
        try:
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
            from lerobot.robots.so_follower.so_follower import SOFollower

            config = SOFollowerRobotConfig(
                port=self.port,
                id=self.robot_id,
            )
            self.robot = SOFollower(config)
            self.robot.connect()
            self.connected = True
            print(f"[ARM] Connected to SO-101 on {self.port}")
        except ImportError:
            print("[ARM] ERROR: lerobot not installed. Install with:")
            print("      pip install lerobot[feetech]")
            print("[ARM] Running in PREVIEW MODE")
        except Exception as e:
            print(f"[ARM] ERROR connecting: {e}")
            print("[ARM] Running in PREVIEW MODE")

    def _map_value(self, normalized, joint_name):
        """Map a normalized [0.0, 1.0] value to the joint's degree range."""
        lo, hi, center = self.joint_ranges[joint_name]
        return center + lo + normalized * (hi - lo)

    def compute_joint_positions(self, hand_x, hand_y, hand_z, hand_tilt, hand_roll):
        """Convert hand X, Y, Z, tilt, roll (all [0.0, 1.0]) to joint angles.

        Mapping:
            hand_x    -> shoulder_pan   (pan left/right)
            hand_y    -> shoulder_lift  (raise/lower)
            hand_z    -> elbow_flex     (extend/retract)
            hand_tilt -> wrist_flex     (tilt hand up/down)
            hand_roll -> wrist_roll     (rotate hand)
        """
        positions = {
            "shoulder_pan":  self._map_value(hand_x, "shoulder_pan"),
            "shoulder_lift": self._map_value(hand_y, "shoulder_lift"),
            "elbow_flex":    self._map_value(hand_z, "elbow_flex"),
            "wrist_flex":    self._map_value(hand_tilt, "wrist_flex"),
            "wrist_roll":    self._map_value(hand_roll, "wrist_roll"),
        }
        return positions

    def send_arm_position(self, joint_positions, gripper_value=None):
        """Send joint positions + optional gripper to the arm.
        
        Args:
            joint_positions: dict of {joint_name: degrees}
            gripper_value: 0.0 (closed) to 100.0 (open), or None to keep unchanged
        """
        action = {f"{name}.pos": val for name, val in joint_positions.items()}
        if gripper_value is not None:
            action["gripper.pos"] = gripper_value

        if self.connected and self.robot:
            try:
                self.robot.send_action(action)
            except Exception as e:
                print(f"[ARM] Error: {e}")
        return action

    def set_gripper(self, state):
        """Set gripper: 'open' or 'close'. Returns True if command was new."""
        if state == self.last_gripper_cmd:
            return False
        self.last_gripper_cmd = state
        return True

    def get_gripper_value(self):
        """Get current gripper target value."""
        if self.last_gripper_cmd == "open":
            return 100.0
        elif self.last_gripper_cmd == "close":
            return 0.0
        return None

    def disconnect(self):
        if self.connected and self.robot:
            try:
                self.robot.disconnect()
                print("[ARM] Disconnected.")
            except Exception as e:
                print(f"[ARM] Error disconnecting: {e}")


# ============================================================
# Drawing Helpers
# ============================================================

def draw_position_bars(frame, x, y, z, tilt, roll, start_y=110, bar_w=150, bar_h=16):
    """Draw X/Y/Z/Tilt/Roll indicator bars on the frame."""
    labels = [("X", x, (66, 133, 244)),   # Blue
              ("Y", y, (52, 168, 83)),     # Green
              ("Z", z, (234, 67, 53)),     # Red
              ("Tilt", tilt, (255, 165, 0)), # Orange
              ("Roll", roll, (128, 0, 128))] # Purple
    
    for i, (label, val, color) in enumerate(labels):
        oy = start_y + i * (bar_h + 8)
        # Background bar
        cv2.rectangle(frame, (20, oy), (20 + bar_w, oy + bar_h), (50, 50, 50), -1)
        # Filled portion
        fill_w = int(val * bar_w)
        cv2.rectangle(frame, (20, oy), (20 + fill_w, oy + bar_h), color, -1)
        # Border
        cv2.rectangle(frame, (20, oy), (20 + bar_w, oy + bar_h), (200, 200, 200), 1)
        # Label
        cv2.putText(frame, f"{label}: {val:.2f}", (bar_w + 30, oy + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def draw_joint_values(frame, joint_positions, gripper_val, x_offset=None, start_y=110):
    """Draw current joint angle values on the right side of the frame."""
    h, w = frame.shape[:2]
    if x_offset is None:
        x_offset = w - 250

    cv2.putText(frame, "JOINT ANGLES", (x_offset, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    items = list(joint_positions.items())
    if gripper_val is not None:
        items.append(("gripper", gripper_val))

    for i, (name, val) in enumerate(items):
        oy = start_y + 15 + i * 20
        short_name = name.replace("shoulder_", "shldr_").replace("wrist_", "wrst_")
        if name == "gripper":
            text = f"{short_name}: {val:.0f}%"
        else:
            text = f"{short_name}: {val:+.1f} deg"
        cv2.putText(frame, text, (x_offset, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ============================================================
# Camera Auto-Detection
# ============================================================

def auto_detect_camera():
    """Scan available cameras and pick the best one.
    
    Strategy: pick the HIGHEST available index, which is typically the
    external USB webcam (index 0 is usually the built-in laptop camera).
    """
    print("[CAM] Scanning for cameras...")
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  Camera {i}: {w}x{h} @ {fps:.0f}fps")
            available.append(i)
            cap.release()

    if not available:
        print("[CAM] ERROR: No cameras found!")
        return 0

    if len(available) == 1:
        chosen = available[0]
        print(f"[CAM] Only one camera found, using camera {chosen}")
    else:
        # Pick highest index (likely the external USB webcam)
        chosen = available[-1]
        print(f"[CAM] Multiple cameras found, using camera {chosen} (likely USB webcam)")
        print(f"[CAM] To override: --camera {available[0]}")

    return chosen


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hand tracking -> SO-101 arm control")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port for SO-101 (e.g. COM3 or /dev/ttyACM0). "
                             "Omit for preview mode.")
    parser.add_argument("--robot-id", type=str, default="gesture_follower",
                        help="Robot calibration ID (default: gesture_follower)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index. If omitted, auto-detects USB webcam.")
    parser.add_argument("--smoothing", type=float, default=0.4,
                        help="Position smoothing (0=none, 0.7=heavy). Default: 0.4")
    parser.add_argument("--pan-range", type=float, default=45.0,
                        help="Shoulder pan range in degrees (default: 45)")
    parser.add_argument("--lift-range", type=float, default=30.0,
                        help="Shoulder lift range in degrees (default: 30)")
    parser.add_argument("--elbow-range", type=float, default=30.0,
                        help="Elbow flex range in degrees (default: 30)")
    args = parser.parse_args()

    # --- Auto-detect camera ---
    camera_idx = args.camera
    if camera_idx is None:
        camera_idx = auto_detect_camera()

    # --- MediaPipe Setup ---
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # --- Arm Setup ---
    arm = ArmController(port=args.port, robot_id=args.robot_id)
    arm.joint_ranges["shoulder_pan"]  = (-args.pan_range, args.pan_range, 0.0)
    arm.joint_ranges["shoulder_lift"] = (-args.lift_range, args.lift_range, 0.0)
    arm.joint_ranges["elbow_flex"]    = (-args.elbow_range, args.elbow_range, 0.0)

    if args.port:
        arm.connect()
    else:
        print("[ARM] No --port specified. Running in PREVIEW MODE.")
        print("[ARM] To connect: python main.py --port COM3")

    # --- Hand Tracker ---
    tracker = HandPositionTracker(smoothing=args.smoothing)

    # --- Camera ---
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_idx}")
        return
    print(f"[CAM] Using camera index {camera_idx}")

    print("\n" + "=" * 55)
    print("  HAND TRACKING -> SO-101 ARM CONTROL")
    print("  Move hand to control arm (X, Y, Z)")
    print("  Open hand  -> Opens gripper")
    print("  Closed fist -> Closes gripper")
    print("  Press 'r' to recalibrate Z-depth")
    print("  Press 'q' to quit")
    print("=" * 55 + "\n")

    # Colors
    COLOR_OPEN = (0, 255, 0)
    COLOR_FIST = (0, 0, 255)
    COLOR_OTHER = (255, 165, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_YELLOW = (0, 255, 255)

    # Gesture debouncing
    gesture_hold_time = 0.3
    last_gesture = None
    gesture_start_time = None
    confirmed_gesture = None

    # Position tracking state
    hand_x, hand_y, hand_z, hand_tilt, hand_roll = 0.5, 0.5, 0.5, 0.5, 0.5
    joint_positions = arm.compute_joint_positions(0.5, 0.5, 0.5, 0.5, 0.5)
    tracking_active = False
    frames_since_lost = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_gesture = None
        fingers_count = 0

        if results.multi_hand_landmarks:
            tracking_active = True
            frames_since_lost = 0

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw skeleton
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Position tracking (X, Y, Z, Tilt, Roll) ---
                hand_x, hand_y, hand_z, hand_tilt, hand_roll = tracker.update(hand_landmarks)

                # --- Compute arm joint positions ---
                joint_positions = arm.compute_joint_positions(hand_x, hand_y, hand_z, hand_tilt, hand_roll)

                # --- Detect gesture for gripper ---
                current_gesture, fingers_count = detect_gesture(hand_landmarks)

                # --- Compute bounding box ---
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = max(0, int(min(x_coords) * w) - 20)
                y_min = max(0, int(min(y_coords) * h) - 20)
                x_max = min(w, int(max(x_coords) * w) + 20)
                y_max = min(h, int(max(y_coords) * h) + 20)

                # Color by gesture
                if current_gesture == "OPEN":
                    color = COLOR_OPEN
                elif current_gesture == "FIST":
                    color = COLOR_FIST
                else:
                    color = COLOR_OTHER

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Handedness label
                hand_label = "Hand"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[hand_idx].classification[0].label

                label = f"{hand_label}: {current_gesture} ({fingers_count}/5)"
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Palm center dot
                palm = hand_landmarks.landmark[9]
                cx, cy = int(palm.x * w), int(palm.y * h)
                cv2.circle(frame, (cx, cy), 10, color, cv2.FILLED)

                # Draw crosshair at palm center
                cv2.line(frame, (cx - 20, cy), (cx + 20, cy), COLOR_CYAN, 1)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), COLOR_CYAN, 1)

                # XYZ text near palm
                cv2.putText(frame, f"({hand_x:.2f}, {hand_y:.2f}, {hand_z:.2f})",
                            (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CYAN, 1)

        else:
            # Hand lost
            frames_since_lost += 1
            if frames_since_lost > 15:  # ~0.5s at 30fps
                tracking_active = False
                tracker.reset()

        # --- Gesture debouncing ---
        if current_gesture != last_gesture:
            last_gesture = current_gesture
            gesture_start_time = time.time()
        elif current_gesture and gesture_start_time:
            hold_duration = time.time() - gesture_start_time
            if hold_duration >= gesture_hold_time:
                if current_gesture != confirmed_gesture:
                    confirmed_gesture = current_gesture
                    if current_gesture == "OPEN":
                        arm.set_gripper("open")
                    elif current_gesture == "FIST":
                        arm.set_gripper("close")

        # --- Send to arm (only when tracking is active) ---
        gripper_val = arm.get_gripper_value()
        if tracking_active:
            action = arm.send_arm_position(joint_positions, gripper_val)

        # ============== HUD ==============

        # Gripper state
        state_text = "GRIPPER: "
        if confirmed_gesture == "OPEN":
            state_text += "OPEN"
            state_color = COLOR_OPEN
        elif confirmed_gesture == "FIST":
            state_text += "CLOSED"
            state_color = COLOR_FIST
        else:
            state_text += "---"
            state_color = COLOR_WHITE

        cv2.putText(frame, state_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        # Connection + tracking status
        conn_text = f"ARM: {'CONNECTED' if arm.connected else 'PREVIEW'}"
        conn_color = COLOR_OPEN if arm.connected else COLOR_OTHER
        cv2.putText(frame, conn_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1)

        track_text = f"TRACKING: {'ACTIVE' if tracking_active else 'LOST'}"
        track_color = COLOR_OPEN if tracking_active else COLOR_FIST
        cv2.putText(frame, track_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)

        # X, Y, Z, Tilt, Roll bars
        if tracking_active:
            draw_position_bars(frame, hand_x, hand_y, hand_z, hand_tilt, hand_roll, start_y=100)
            draw_joint_values(frame, joint_positions, gripper_val)

        # Bottom bar: instructions
        cv2.putText(frame, "'q' quit | 'r' recalibrate Z",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow("Hand Tracking -> SO-101 Arm", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.ref_palm_size = None
            print("[TRACKER] Z-depth reference recalibrated")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    arm.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()