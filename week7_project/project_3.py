import cv2
import numpy as np
from dorothy import Dorothy

dot = Dorothy(width=1280, height=780)

class AntiSurveillanceMirror:

    def __init__(self, cam_index=0):
        # Previous grayscale frame
        self.prev_gray = None
        # Store face-center trail positions
        self.trail_points = []
        self.max_trail_len = 30

        # Motion thresholds for different visual modes
        self.motion_low = 5.0
        self.motion_high = 15.0

        # Fake “data leak” text system
        self.leak_lines = []
        self.leak_frames_left = 20
        self.leak_lifetime = 100000000

        self.cam_index = cam_index
        # Start Dorothy render loop
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        # Set up camera
        self.camera = cv2.VideoCapture(4)

        # Haar-cascade face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def draw(self):
        # Pull camera frame
        success, frame = self.camera.read()
        if not success:
            return

        # Convert to RGB and resize to Dorothy canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (dot.width, dot.height))

        # Grayscale for detection + motion analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        # Background-subtraction style motion measurement
        motion_level = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion_level = np.mean(diff)
        self.prev_gray = gray.copy()

        if len(faces) == 0:
            mode = "idle"
        else:
            if motion_level < self.motion_low:
                mode = "pixelate"   # Low motion → soft anonymisation
            elif motion_level < self.motion_high:
                mode = "data_leak"     # Medium motion → data-leak effect
            else:
                mode = "vanish"     # High motion → hard pixelation

        # Apply visual effects to each detected face
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2

            # Store trail of face center
            self.trail_points.append((cx, cy))
            if len(self.trail_points) > self.max_trail_len:
                self.trail_points.pop(0)

            u = cx / dot.width
            v = cy / dot.height

            # Mode-based filtering
            if mode == "pixelate":
                self.apply_feature_pixelate(frame, x, y, w, h, u, v, strong=False)
            elif mode == "data_leak":
                self.apply_data_leak(frame, x, y, w, h, u, v)
            elif mode == "vanish":
                self.apply_feature_pixelate(frame, x, y, w, h, u, v, strong=True)
            else:
                # Idle mode: simple detection box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw face-movement trail
        self.draw_trail(frame)

        # Display current mode + motion reading
        text = f"mode: {mode} | motion: {motion_level:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Output final frame to the Dorothy canvas
        dot.canvas = frame

    def pixelate_region(self, img, x1, y1, x2, y2, block_size):
        # Basic block-pixelation
        h = y2 - y1
        w = x2 - x1
        if h <= 0 or w <= 0:
            return

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        roi = img[y1:y2, x1:x2]
        h_roi, w_roi = roi.shape[:2]
        if h_roi <= 0 or w_roi <= 0:
            return

        bs = max(2, block_size)
        h_small = max(1, h_roi // bs)
        w_small = max(1, w_roi // bs)

        temp = cv2.resize(roi, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = pixelated

    def apply_feature_pixelate(self, img, x, y, w, h, u, v, strong=False):
        # Block-size varies with face position
        base_block = 16 if strong else 8
        block_size = int(base_block + u * (base_block * 1.5))
        block_size = max(4, min(block_size, 50))

        # Pixelate regions（eyes, nose, mouth)
        eye_y1 = int(y + 0.20 * h)
        eye_y2 = int(y + 0.40 * h)
        self.pixelate_region(img, x, eye_y1, x + w, eye_y2, block_size)

        nose_y1 = int(y + 0.40 * h)
        nose_y2 = int(y + 0.65 * h)
        self.pixelate_region(img, x, nose_y1, x + w, nose_y2, block_size)

        mouth_y1 = int(y + 0.65 * h)
        mouth_y2 = int(y + 0.90 * h)
        self.pixelate_region(img, x, mouth_y1, x + w, mouth_y2, block_size)

        cv2.rectangle(img, (x, y), (x + w, y + h),
                      (255, 255, 255) if not strong else (0, 0, 0), 1)

    def apply_data_leak(self, img, x, y, w, h, u, v):
        # Blur the face like privacy filters
        face_roi = img[y:y + h, x:x + w]
        blurred = cv2.GaussianBlur(face_roi, (15, 15), 0)
        img[y:y + h, x:x + w] = blurred

        if self.leak_frames_left <= 0 or not self.leak_lines:
            fake_id = f"ID: {np.random.randint(100000, 999999)}"
            fake_age = f"AGE: {np.random.randint(18, 60)}"
            fake_emo = np.random.choice(["EMO: HAPPY", "EMO: NEUTRAL", "EMO: ANGRY"])
            fake_risk = np.random.choice(["RISK: LOW", "RISK: MED", "RISK: HIGH"])

            self.leak_lines = [fake_id, fake_age, fake_emo, fake_risk]
            self.leak_frames_left = self.leak_lifetime
        else:
            self.leak_frames_left -= 1

        # Draw text beside the face
        text_x = x + w + 10
        text_y = y + 15
        if text_x > img.shape[1] - 150:
            text_x = x - 150

        for i, line in enumerate(self.leak_lines):
            ty = text_y + i * 18
            cv2.putText(
                img,
                line,
                (text_x, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_trail(self, img):
        # Connect previous face positions to show movement over time
        if len(self.trail_points) < 2:
            return

        for i in range(1, len(self.trail_points)):
            x1, y1 = self.trail_points[i - 1]
            x2, y2 = self.trail_points[i]

            # Fade from dark to bright
            alpha = i / len(self.trail_points)
            color_val = int(50 + alpha * 205)
            color = (color_val, color_val, color_val)

            cv2.line(img, (x1, y1), (x2, y2), color, 2)


AntiSurveillanceMirror(cam_index=0)
