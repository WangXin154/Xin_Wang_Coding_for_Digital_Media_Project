## Week 7 — Anti-Surveillance Mirror（project_3.py）
**An interactive video experiment on surveillance visibility, body data and counter-surveillance strategies**

## 1. Project Introduction: Make an 'Anti-Surveillance' Mirror

This week’s session focused on how video systems detect, classify, and interpret human bodies through techniques such as background subtraction, motion detection, face detection, and activity recognition. These methods—often perceived as neutral—carry embedded biases and political implications, as seen in issues such as racial bias in facial recognition or the pseudoscience of emotion classification.

In response, I designed an **Anti-Surveillance Mirror**: an interactive system that reverses the *logic of surveillance*. Instead of identifying the user, it disrupts, obscures, or hides the very information surveillance systems seek to extract. Depending on the user’s movement intensity, the mirror transitions through several “defensive modes”:

- `idle`: minimal augmentation

- `pixelate`: distortion of facial features (anti–face recognition)
- `data_leak`: display of fabricated personal metrics (satire of biometric scoring)
- `vanish`: full obfuscation triggered by rapid motion (anti-tracking)

The system stages a negotiation of *visibility* between the user and the algorithm.

---

## 2. Core Technical Components

**2.1 Video Frame Processing**
Each frame is captured, converted to RGB, and resized:
```python
success, frame = self.camera.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (dot.width, dot.height))
```
This reinforces the idea that *video = a sequence of discrete images*, each subject to computational inspection.

**2.2 Face Detection（Haar Cascade）**

```python
self.face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )
```

This module identifies the user’s face—a key element of contemporary surveillance technologies.

**2.3 Motion Detection（Temporal Difference）**

```python
diff = cv2.absdiff(gray, self.prev_gray)
motion_level = np.mean(diff)
```
The **mean frame difference** functions as an approximate measure of bodily intensity. This value controls transitions between defensive states.

**2.4 Interaction Model as Critique**

Surveillance systems often encode:
- racial or gender bias
- behavioural suspicion (“abnormal behaviour” detection)
- reductive emotion models
Recognising these issues, I designed each interaction state to conceptually push back against the logic of visibility and classification. The system does not “recognise” the user; instead, it **misrecognises on purpose**.

---

## 3. System Logic: Four Anti-Surveillance Modes
**3.1 Idle Mode — subtle border**

If no face is detected, the system enters `idle`. The lens only provides the most basic view and does not perform analysis.

**3.2 `pixelate`Mode - targeted obfuscation**

When a face is detected with moderate motion, specific facial regions (eyes, nose, mouth) are pixelated:
```python
self.pixelate_region(img, x, eye_y1, x+w, eye_y2, block_size)
```

This references artistic and activist counter-surveillance techniques aimed at disrupting biometric systems.

<img width="1042" alt="week7_1" src="https://git.arts.ac.uk/user-attachments/assets/576561f7-2799-4257-8e43-621e535c4cd4" />


https://git.arts.ac.uk/user-attachments/assets/2b11c06b-528f-4e43-9462-e6d7e19821a3


**3.3 `data_leak` Mode - fabricated biometric profiling**

Triggered by medium motion intensity, the system overlays **fake identity and emotion labels**:

```python
        if self.leak_frames_left <= 0 or not self.leak_lines:
            fake_id = f"ID: {np.random.randint(100000, 999999)}"
            fake_age = f"AGE: {np.random.randint(18, 60)}"
            fake_emo = np.random.choice(["EMO: HAPPY", "EMO: NEUTRAL", "EMO: ANGRY"])
            fake_risk = np.random.choice(["RISK: LOW", "RISK: MED", "RISK: HIGH"])

```

This mode satirises:
- “risk scoring”
- emotion recognition
- identity prediction

by demonstrating how arbitrary and absurd these classifications can be.

<img width="837" alt="week7_2" src="https://git.arts.ac.uk/user-attachments/assets/4663f1e9-d85c-48ba-9169-08304a414ceb" />

This is actually a satire on the absurdity of automatically judging risk/emotion/identity in surveillance systems.

**3.4 Vanish Mode — full-body obfuscation**

High-intensity movement triggers:
- aggressive full-face pixelation
- rapid motion trails
- near-complete loss of trackability

This simulates the breakdown point of surveillance systems:
**If the system cannot stabilise you, it cannot classify you.**

https://git.arts.ac.uk/user-attachments/assets/ffa3adb4-17f6-4b88-b354-786569b80c28

---

## 4.Visualisation: Motion Trajectory

The system records and draws the user’s movement path:
```python
self.trail_points.append((cx, cy))
cv2.line(img, (x1, y1), (x2, y2), color, 2)

```
In typical surveillance, motion trails function as behavioural logs; here they become **user-owned marks**, emphasising agency rather than extraction.

---

## 5. Run Instructions
**5.1 Dependencies**
```python
pip install numpy opencv-python pillow dorothy
```
**5.2 Directory structure**
```python
week7_project/
    main.py
    README.md
```
**5.3 run**
`project_3.py`

A Dorothy window will open and you will see:

Camera real-time picture

Your face is 'anti-surveilled' in various ways

Three different monitoring and defense modes switch when you move

---

## 6. Reflection: Who has the right to “see”?

Through Week7’s courses and this project, I realized:

In the surveillance system, we are never 'seen neutrally', but are classified, archived, and interpreted.

Face detection, emotion recognition, and activity recognition may seem like technologies, but they all have huge political and social consequences.

My Anti-Surveillance Mirror attempts to give users the experience of:

- The tension of being monitored

- Anti-surveillance initiative

- Manipulation of visual information

It’s not a privacy tool, but a mirror that 'makes you aware of how you are seen.'
