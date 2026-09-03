import socket
import json
import time
import os
import glob
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat

# ===== CONFIG =====
UDP_IP = "127.0.0.1"
UDP_PORT = 5052
GESTURES_FILE = "gestures.json"
TRAINING_DIR = "training_data"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ===== MEDIAPIPE =====
print("Loading model...")
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)
print("Model OK!")

# ===== GESTURES =====
# Format: gestures["a"] = {"type":"pose", "data": [21 landmarks]}
#         gestures["привет"] = {"type":"anim", "data": [frame1, frame2, ...]}
def load_gestures():
    if os.path.exists(GESTURES_FILE):
        with open(GESTURES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert old format (raw list) to new format
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = {"type": "pose", "data": v}
        return data
    return {}

def save_gestures(g):
    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=2, ensure_ascii=False)

gestures = load_gestures()

# ===== SPEECH =====
def init_speech():
    try:
        from vosk import Model, KaldiRecognizer
        import sounddevice as sd
        import queue
        if not os.path.exists("vosk-model-small-ru-0.22"):
            return None
        model = Model("vosk-model-small-ru-0.22")
        rec = KaldiRecognizer(model, 16000)
        q = queue.Queue()
        def cb(indata, frames, t, status):
            q.put(bytes(indata))
        stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=cb)
        stream.start()
        return {"rec": rec, "stream": stream, "q": q}
    except Exception as e:
        print(f"Speech error: {e}")
        return None

speech = init_speech()
if speech:
    print("Speech ON!")

# ===== EXTRACT =====
def from_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    r = detector.detect(mp_img)
    if r.hand_landmarks and len(r.hand_landmarks) > 0:
        return [{"x": l.x, "y": l.y, "z": l.z} for l in r.hand_landmarks[0]]
    return None

def frames_from_video(path, sample_interval=1, max_frames=40, max_seconds=2.0):
    """Extract hand frames from video (first 2 seconds only), resampled to max_frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frame = int(max_seconds * fps) if fps > 0 else 60

    frames = []
    prev_landmarks = None
    i = 0

    while i < max_frame:
        ok, img = cap.read()
        if not ok:
            break

        if i % sample_interval == 0:
            lm = from_image(img)
            if lm:
                if prev_landmarks:
                    diff = sum(
                        abs(lm[j]["x"] - prev_landmarks[j]["x"]) +
                        abs(lm[j]["y"] - prev_landmarks[j]["y"]) +
                        abs(lm[j]["z"] - prev_landmarks[j]["z"])
                        for j in range(21)
                    )
                    if diff < 0.05:
                        i += 1
                        continue

                frames.append(lm)
                prev_landmarks = lm
        i += 1

    cap.release()

    if len(frames) == 0:
        return []

    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(j * step)] for j in range(max_frames)]

    print(f"    Extracted {len(frames)} frames (first {max_seconds}s)")
    return frames

def best_from_image(img):
    return from_image(img)

# ===== ANIMATION =====
NEUTRAL = [
    {"x":0.50,"y":0.85,"z":0.00},
    {"x":0.40,"y":0.72,"z":0.02},
    {"x":0.33,"y":0.60,"z":0.03},
    {"x":0.27,"y":0.50,"z":0.02},
    {"x":0.22,"y":0.42,"z":0.00},
    {"x":0.38,"y":0.42,"z":0.00},
    {"x":0.35,"y":0.28,"z":-0.01},
    {"x":0.33,"y":0.18,"z":-0.02},
    {"x":0.32,"y":0.10,"z":-0.03},
    {"x":0.44,"y":0.40,"z":0.00},
    {"x":0.44,"y":0.26,"z":-0.01},
    {"x":0.44,"y":0.16,"z":-0.02},
    {"x":0.44,"y":0.08,"z":-0.03},
    {"x":0.50,"y":0.42,"z":0.00},
    {"x":0.52,"y":0.28,"z":-0.01},
    {"x":0.54,"y":0.18,"z":-0.02},
    {"x":0.55,"y":0.10,"z":-0.03},
    {"x":0.56,"y":0.44,"z":0.00},
    {"x":0.60,"y":0.32,"z":-0.01},
    {"x":0.63,"y":0.24,"z":-0.02},
    {"x":0.65,"y":0.18,"z":-0.03},
]

FINGER_GROUPS = {
    "thumb":  [1,2,3,4],
    "index":  [5,6,7,8],
    "middle": [9,10,11,12],
    "ring":   [13,14,15,16],
    "pinky":  [17,18,19,20],
}

def ease(t):
    return 1 - (1 - t) ** 3

def lerp_lm(a, b, t):
    """Interpolate between two landmark frames"""
    result = []
    for i in range(min(len(a), len(b))):
        result.append({
            "x": a[i]["x"] + (b[i]["x"] - a[i]["x"]) * t,
            "y": a[i]["y"] + (b[i]["y"] - a[i]["y"]) * t,
            "z": a[i]["z"] + (b[i]["z"] - a[i]["z"]) * t,
        })
    return result

def send(lm, gesture_name=""):
    pk = {
        "landmarks": lm,
        "right_hand_detected": True,
        "left_hand_detected": False,
    }
    if gesture_name:
        pk["gesture_name"] = gesture_name
    sock.sendto(json.dumps(pk).encode(), (UDP_IP, UDP_PORT))

def smooth_transition(from_lm, to_lm, steps=20, delay=0.012):
    """Smooth transition between two poses with finger stagger"""
    for f in range(steps + 1):
        t = f / steps
        # Smoother easing: cubic ease-in-out
        if t < 0.5:
            t_ease = 4 * t * t * t
        else:
            t_ease = 1 - (-2 * t + 2) ** 3 / 2

        frame = [{} for _ in range(21)]

        # Wrist moves first, smoothly
        frame[0] = {
            "x": from_lm[0]["x"] + (to_lm[0]["x"] - from_lm[0]["x"]) * t_ease,
            "y": from_lm[0]["y"] + (to_lm[0]["y"] - from_lm[0]["y"]) * t_ease,
            "z": from_lm[0]["z"] + (to_lm[0]["z"] - from_lm[0]["z"]) * t_ease,
        }

        # Palm (landmarks 5,9,13,17) follows wrist
        palm_indices = [5, 9, 13, 17]
        t_palm = max(0, min(1, t_ease * 1.2))
        for idx in palm_indices:
            frame[idx] = {
                "x": from_lm[idx]["x"] + (to_lm[idx]["x"] - from_lm[idx]["x"]) * t_palm,
                "y": from_lm[idx]["y"] + (to_lm[idx]["y"] - from_lm[idx]["y"]) * t_palm,
                "z": from_lm[idx]["z"] + (to_lm[idx]["z"] - from_lm[idx]["z"]) * t_palm,
            }

        # Each finger: knuckle -> tip, with natural delay
        delays = {"thumb":0.05, "index":0.08, "middle":0.12, "ring":0.16, "pinky":0.20}
        for fname, indices in FINGER_GROUPS.items():
            d = delays[fname]
            # Progress for this finger (0 to 1, accounting for delay)
            t_finger = max(0, min(1, (t_ease - d) / (1 - d)))
            # Smooth each joint: MCP -> PIP -> DIP -> tip
            for j, idx in enumerate(indices):
                # Each joint has slight additional delay
                t_joint = max(0, min(1, (t_finger - j * 0.03) / (1 - j * 0.03)))
                t_joint = t_joint * t_joint * (3 - 2 * t_joint)  # smoothstep
                frame[idx] = {
                    "x": from_lm[idx]["x"] + (to_lm[idx]["x"] - from_lm[idx]["x"]) * t_joint,
                    "y": from_lm[idx]["y"] + (to_lm[idx]["y"] - from_lm[idx]["y"]) * t_joint,
                    "z": from_lm[idx]["z"] + (to_lm[idx]["z"] - from_lm[idx]["z"]) * t_joint,
                }

        send(frame)
        time.sleep(delay)

def play_gesture(gesture_data, hold=0.8, name=""):
    """Play a gesture - handles both pose and animation"""
    if isinstance(gesture_data, dict):
        gtype = gesture_data.get("type", "pose")
        frames = gesture_data.get("data", [])
    else:
        gtype = "pose"
        frames = gesture_data

    # Send gesture name for display
    if name:
        send(NEUTRAL, gesture_name=name)

    if gtype == "pose" or len(frames) == 0:
        target = frames[0] if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], dict) else frames
        smooth_transition(NEUTRAL, target, steps=30, delay=0.010)
        time.sleep(hold)
        smooth_transition(target, NEUTRAL, steps=30, delay=0.010)

    elif gtype == "anim":
        smooth_transition(NEUTRAL, frames[0], steps=20, delay=0.006)
        time.sleep(0.05)

        for i in range(len(frames) - 1):
            smooth_transition(frames[i], frames[i+1], steps=3, delay=0.006)

        time.sleep(hold)
        smooth_transition(frames[-1], NEUTRAL, steps=20, delay=0.006)

# ===== COMMANDS =====
def cmd_auto_train():
    print("\n--- AUTO-TRAIN ---")
    print("  Images -> single pose gesture")
    print("  Videos -> animation sequence\n")

    exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.mp4","*.avi","*.mov","*.mkv","*.webm"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(TRAINING_DIR, e)))

    if not files:
        print("  Folder training_data/ is empty!")
        return

    print(f"  Found {len(files)} files\n")
    ok = 0
    for i, fp in enumerate(sorted(files)):
        fn = os.path.basename(fp)
        name = os.path.splitext(fn)[0].lower()
        pct = int((i/len(files))*100)
        print(f"  [{pct}%] {fn} -> '{name}' ... ", end="", flush=True)

        is_vid = any(fn.lower().endswith(v) for v in [".mp4",".avi",".mov",".mkv",".webm"])

        if is_vid:
            frames = frames_from_video(fp, sample_interval=1)  # every frame
            if len(frames) >= 2:
                gestures[name] = {"type": "anim", "data": frames}
                ok += 1
                print(f"OK (animation: {len(frames)} frames)")
            elif len(frames) == 1:
                gestures[name] = {"type": "pose", "data": frames[0]}
                ok += 1
                print("OK (single frame)")
            else:
                print("SKIP (no hands)")
        else:
            img = cv2.imread(fp)
            if img is None:
                print("SKIP (can't read)")
                continue
            lm = best_from_image(img)
            if lm:
                gestures[name] = {"type": "pose", "data": lm}
                ok += 1
                print("OK (pose)")
            else:
                print("SKIP (no hand)")

    save_gestures(gestures)
    print(f"\n  Done! {ok}/{len(files)} learned")
    print(f"  Total: {len(gestures)} gestures\n")

def cmd_show():
    if not gestures:
        print("  No gestures saved")
        return
    print(f"\n  Saved gestures ({len(gestures)}):")
    for k in sorted(gestures.keys()):
        g = gestures[k]
        if isinstance(g, dict):
            t = g.get("type", "?")
            n = len(g.get("data", []))
            print(f"    {k}  [{t}: {n} frames]")
        else:
            print(f"    {k}  [pose]")
    print()

def cmd_delete():
    letter = input("  Letter to delete: ").strip().lower()
    if letter in gestures:
        del gestures[letter]
        save_gestures(gestures)
        print(f"  Deleted: {letter}")
    else:
        print(f"  Not found: {letter}")

def cmd_train_live():
    print("\n--- LIVE TRAIN ---")
    letter = input("  Letter/word to record: ").strip().lower()
    if not letter:
        print("  Cancelled")
        return

    mode = input("  [P]ose (one shot) or [A]nimation (record movement)? ").strip().lower()

    print(f"  Opening camera for '{letter}'...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Cannot open camera!")
        return

    if mode == "a":
        # Animation mode: record continuous frames
        print("  Move your hand to perform the gesture...")
        print("  Press ENTER to start recording, ENTER again to stop, ESC to cancel")

        # Wait for first ENTER
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            lm = from_image(img)
            status = "HAND FOUND" if lm else "NO HAND"
            cv2.putText(img, f"Animation: {letter}  [{status}]", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(img, "Press ENTER to START recording", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Train", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                print("  Cancelled")
                return
            elif key == 13:
                break

        # Record frames
        print("  Recording... move your hand!")
        frames = []
        prev = None
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            lm = from_image(img)
            if lm:
                # Skip duplicate frames
                if prev:
                    diff = sum(
                        abs(lm[j]["x"] - prev[j]["x"]) +
                        abs(lm[j]["y"] - prev[j]["y"]) +
                        abs(lm[j]["z"] - prev[j]["z"])
                        for j in range(21)
                    )
                    if diff < 0.02:
                        continue
                frames.append(lm)
                prev = lm
            cv2.putText(img, f"REC: {len(frames)} unique frames", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Train", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == 27:  # ENTER or ESC to stop
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(frames) >= 2:
            gestures[letter] = {"type": "anim", "data": frames}
            save_gestures(gestures)
            print(f"  Saved animation '{letter}' ({len(frames)} frames)!")
        else:
            print("  Need at least 2 frames for animation")

    else:
        # Pose mode: single shot - capture 5 frames and average
        print("  Show your gesture. ENTER=capture, need 5 frames to average.")
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            lm = from_image(img)
            if lm:
                cv2.putText(img, f"Pose: {letter}  Frames: {len(frames)}/5", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for l in lm:
                    h, w, _ = img.shape
                    cv2.circle(img, (int(l["x"]*w), int(l["y"]*h)), 3, (0,255,0), -1)
            else:
                cv2.putText(img, "No hand...", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Train", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 13 and lm:
                frames.append(lm)
                print(f"  Captured {len(frames)}")
                if len(frames) >= 5:
                    avg = []
                    for j in range(21):
                        avg.append({
                            "x": sum(f[j]["x"] for f in frames) / len(frames),
                            "y": sum(f[j]["y"] for f in frames) / len(frames),
                            "z": sum(f[j]["z"] for f in frames) / len(frames),
                        })
                    gestures[letter] = {"type": "pose", "data": avg}
                    save_gestures(gestures)
                    print(f"  Saved pose '{letter}' (averaged {len(frames)} frames)!")
                    break

        cap.release()
        cv2.destroyAllWindows()

def cmd_listen():
    if not speech:
        print("  Speech not available!")
        return

    print("\n--- LISTEN MODE ---")
    print(f"  Trained: {', '.join(sorted(gestures.keys())[:15])}")
    print("  Say a letter/word. Ctrl+C to stop.\n")

    last_played = ""
    last_time = 0

    try:
        while True:
            try:
                data = speech["q"].get(timeout=0.1)
            except:
                continue

            # Only process FINAL results, skip partial
            if speech["rec"].AcceptWaveform(data):
                r = json.loads(speech["rec"].Result())
                t = r.get("text", "").strip().lower()
            else:
                continue  # skip partial results

            if not t:
                continue

            # Skip if same word was just played (< 2 seconds)
            now = time.time()
            if t == last_played and now - last_time < 2:
                continue

            print(f"  Heard: '{t}'")

            words = t.split()
            for w in words:
                if w in gestures:
                    print(f"  >> '{w}' - playing!")
                    last_played = w
                    last_time = time.time()
                    play_gesture(gestures[w], name=w)
                    break  # play only first match
                elif len(w) == 1 and w in gestures:
                    print(f"  >> '{w}' - playing!")
                    last_played = w
                    last_time = time.time()
                    play_gesture(gestures[w], name=w)
                    break
    except KeyboardInterrupt:
        print("\n  Stopped")

def cmd_play():
    letter = input("  Letter to play: ").strip().lower()
    if letter in gestures:
        g = gestures[letter]
        t = g.get("type", "pose") if isinstance(g, dict) else "pose"
        print(f"  Playing '{letter}' [{t}]...")
        play_gesture(g, name=letter)
        print("  Done!")
    else:
        print(f"  Not found: {letter}")

# ===== MENU =====
def menu():
    print()
    print("=" * 50)
    print("  GESTURE TRAINER")
    print("=" * 50)
    print("  A - Auto-train (drop files in training_data/)")
    print("  R - Re-train all (resample to 40 frames)")
    print("  T - Train live (camera)")
    print("  L - Listen (voice -> gesture)")
    print("  P - Play a gesture")
    print("  S - Show saved gestures")
    print("  D - Delete gesture")
    print("  Q - Quit")
    print("=" * 50)
    print(f"  Gestures: {len(gestures)}")
    if gestures:
        types = {}
        for k, v in gestures.items():
            t = v.get("type", "?") if isinstance(v, dict) else "pose"
            types[t] = types.get(t, 0) + 1
        parts = [f"{v} {k}" for k, v in types.items()]
        print(f"  Types: {', '.join(parts)}")
    print("=" * 50)

def cmd_resample():
    """Resample all animations to max 40 frames"""
    count = 0
    for name, data in list(gestures.items()):
        if isinstance(data, dict) and data.get("type") == "anim":
            frames = data.get("data", [])
            if len(frames) > 40:
                step = len(frames) / 40
                data["data"] = [frames[int(j * step)] for j in range(40)]
                print(f"  {name}: {len(frames)} -> 40 frames")
                count += 1
    if count > 0:
        save_gestures()
        print(f"\n  Resampled {count} gestures")
    else:
        print("  All gestures already <= 40 frames")

def main():
    menu()
    while True:
        cmd = input("\n> ").strip().lower()
        if cmd == "a":
            cmd_auto_train()
        elif cmd == "r":
            cmd_resample()
        elif cmd == "t":
            cmd_train_live()
        elif cmd == "l":
            cmd_listen()
        elif cmd == "p":
            cmd_play()
        elif cmd == "s":
            cmd_show()
        elif cmd == "d":
            cmd_delete()
        elif cmd == "q":
            break
        else:
            print("  Unknown command")

    sock.close()

if __name__ == "__main__":
    main()
