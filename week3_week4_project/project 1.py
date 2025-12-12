from dorothy import Dorothy
import librosa
import numpy as np
import math
from cv2 import rectangle

WIDTH, HEIGHT = 1200, 600
SR = 22050     # Audio sample rate
BEAT_DURATION = 0.5    # Duration of one step in seconds
FRAMES_PER_STEP = 8    # How many frames each step lasts

S2_GAIN = 0.25
S3_GAIN = 0.25

# Amplitude → motion mapping parameters
AMP_THRESHOLD = 0.02
AMP_SCALE = 160
SHAKES = 8

dot = Dorothy(WIDTH, HEIGHT)
def draw_flower(center, base_radius, variant=0):
    """Draws a stylised flower at a given position and size."""
    cx, cy = center
    r0 = int(base_radius)

    red_petals = (230, 0, 0)
    pink_petals = (255, 150, 200)
    leaf_green = (0, 160, 0)
    leaf_dark = (0, 110, 0)
    center_yellow = (255, 220, 0)

    # Leaf ring around the flower
    leaf_count = 6
    leaf_r = int(r0 * 0.9)
    for i in range(leaf_count):
        angle = 2 * math.pi * i / leaf_count + variant * 0.25
        lx = cx + int(r0 * 1.2 * math.cos(angle))
        ly = cy + int(r0 * 1.2 * math.sin(angle))
        dot.fill(leaf_green if i % 2 == 0 else leaf_dark)
        dot.circle([lx, ly], int(leaf_r * 0.6))

    # Outer petal ring
    outer_count = 7
    outer_r = int(r0 * 0.8)
    for i in range(outer_count):
        angle = 2 * math.pi * i / outer_count
        px = cx + int(r0 * 0.8 * math.cos(angle))
        py = cy + int(r0 * 0.8 * math.sin(angle))
        dot.fill(red_petals)
        dot.circle([px, py], int(outer_r * 0.55))

    # Inner petal ring
    inner_count = 6
    inner_r = int(r0 * 0.55)
    for i in range(inner_count):
        angle = 2 * math.pi * i / inner_count + 0.3
        px = cx + int(r0 * 0.45 * math.cos(angle))
        py = cy + int(r0 * 0.45 * math.sin(angle))
        dot.fill(pink_petals)
        dot.circle([px, py], int(inner_r * 0.6))

    # Flower center
    dot.fill(center_yellow)
    dot.circle([cx, cy], int(r0 * 0.35))


class MySketch:
    def __init__(self):
        self.sr = SR
        self.beat_duration = BEAT_DURATION
        self.beat_samples = int(self.sr * self.beat_duration)

        # Sync visual frame rate to beat structure
        self.frames_per_step = FRAMES_PER_STEP
        dot.fps = int(self.frames_per_step / self.beat_duration)

        # Simple step sequencer: each step triggers one of three samples
        self.sequence = [
            {"sound": "s1"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
            {"sound": "s3"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
            {"sound": "s3"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
            {"sound": "s1"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
        ]

        # Start Dorothy loop
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        # Load three audio samples and resample to a common SR
        self.s1, _ = librosa.load("week3_week4_project/audio/sample_1.WAV", sr=self.sr)
        self.s2, _ = librosa.load("week3_week4_project/audio/sample_2.WAV", sr=self.sr)
        self.s3, _ = librosa.load("week3_week4_project/audio/sample_3.WAV", sr=self.sr)

        # Normalise sample levels so they have similar peak amplitude
        max1 = np.max(np.abs(self.s1))
        max2 = np.max(np.abs(self.s2))
        max3 = np.max(np.abs(self.s3))

        self.s2 = self.s2 * (max1 / (max2 + 1e-6))
        self.s3 = self.s3 * (max1 / (max3 + 1e-6))

        # Schedule each sample on a global timeline based on beat duration
        total_steps = len(self.sequence)
        events_with_sound = []
        for i, event in enumerate(self.sequence):
            sound = event["sound"]
            if sound == "s1":
                src = self.s1
            elif sound == "s2":
                src = self.s2
            elif sound == "s3":
                src = self.s3
            else:
                continue
            start = i * self.beat_samples    # Start time
            end = start + len(src)
            events_with_sound.append((start, end, src))

        # Determine total audio length
        if not events_with_sound:
            total_samples = total_steps * self.beat_samples
        else:
            total_samples = max(end for (start, end, src) in events_with_sound)

        # Mix all scheduled samples into a single audio buffer
        blank_audio = np.zeros(total_samples, dtype=np.float32)
        for (start, end, src) in events_with_sound:
            blank_audio[start:end] += src

        dot.music.start_sample_stream(blank_audio, sr=self.sr, buffer_size=2048)

        # Precompute per-step RMS energy to normalise visual reaction
        self.step_energy = []
        for i in range(total_steps):
            start = i * self.beat_samples
            end = min((i + 1) * self.beat_samples, len(blank_audio))
            window = blank_audio[start:end]
            if len(window) == 0:
                rms = 0.0
            else:
                rms = float(np.sqrt(np.mean(window ** 2)))
            self.step_energy.append(rms)
        mx = max(self.step_energy) if max(self.step_energy) > 0 else 1.0
        self.step_energy = [e / mx for e in self.step_energy]

    def draw(self):
        # Beat-synchronised step index and local frame within the step
        total_steps = len(self.sequence)

        step_index = (dot.frame // self.frames_per_step) % total_steps
        local_frame = dot.frame % self.frames_per_step

        amp = dot.music.amplitude()
        amp_clamped = max(0.0, min(1.0, float(amp)))

        bar_w = 320
        bar_h = 18
        bx = 16
        by = 12
        cover = dot.get_layer()
        # Background of bar
        rectangle(cover, (bx, by), (bx + bar_w, by + bar_h), (30, 30, 30), -1)
        # Filled part based on amplitude
        fill_w = int(bar_w * amp_clamped)
        if fill_w > 0:
            rectangle(cover, (bx, by), (bx + fill_w, by + bar_h), (0, 220, 120), -1)
        dot.draw_layer(cover, 1.0)


        TEMP_THRESHOLD = 0.015
        TEMP_SCALE = 220
        TEMP_SHAKES = 10
        FLOWER_OFFSET_MULT = 1.8

        offset_x, offset_y = 0, 0
        if amp_clamped > TEMP_THRESHOLD:
            mag = int(TEMP_SCALE * (amp_clamped - TEMP_THRESHOLD) / (1.0 - TEMP_THRESHOLD)) + 2
            phase = int((local_frame / self.frames_per_step) * (TEMP_SHAKES * 2))
            direction = -1 if phase % 2 == 0 else 1
            offset_x = direction * mag
            offset_y = int(offset_x * 0.45)

        # Background colour and audio waveform
        dot.background((120, 0, 0))
        dot.draw_waveform(dot.canvas, col=(230, 255, 180), with_playhead=True)

        # Apply motion offset to all flower positions
        ox = int(offset_x * FLOWER_OFFSET_MULT)
        oy = int(offset_y * FLOWER_OFFSET_MULT)

        edge_flowers = [
            [150 + ox, 80 + oy],
            [600 + ox, 80 + oy],
            [1050 + ox, 80 + oy],

            [150 + ox, 520 + oy],
            [600 + ox, 520 + oy],
            [1050 + ox, 520 + oy],
        ]
        for i, c in enumerate(edge_flowers):
            draw_flower(c, base_radius=55, variant=i)

        # Larger flowers at left and right center
        side_centers = [
            [80 + ox, 300 + oy],
            [1120 + ox, 300 + oy],
        ]
        for i, c in enumerate(side_centers):
            draw_flower(c, base_radius=65, variant=10 + i)


MySketch()
