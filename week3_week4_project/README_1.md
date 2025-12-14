# Week 3-4 – Repetitiveness, "Earthy Pop", and Audio - Visual Reconstruction(Project_1.py)
**Project Overview**

In this project, I explored how repetition can function as a structural device in both music and visual media. Using the highly recognisable online song Great Northeast as audio source material, I aimed to investigate two questions:

- **1.Can disassembling and recombining short audio samples create new musical structures?**
- **2.Is it possible to design a visual language that corresponds to the intentionally kitsch aesthetic of the Northeastern floral padded jacket?**

This project allowed me to reinterpret a piece of “earthy pop culture” through computational processes, focusing on rhythmic reconstruction and audio-driven visual behaviour.

**The core idea** of this project:

- Select three different character **samples** from the original song; 
- Arrange them into a **loop sequence**, making **repetition** itself the core of the rhythm; 
- Map the final **energy and amplitude** onto the underlying visuals, causing the flowers to shake in real time to the rhythm, making the video more rhythmic.
# 1.Conception:

- A mini step sequencer  
- A digital restructuring and structure of Northeast folk culture
- An exploration of low-quality aesthetics

---

## 2.Audio: Sample disassembly and reassembly

I extracted three stylistically distinct samples from the song:

- **Sample 1** – exaggerated, humorous pitch-shift gestures
- **Sample 2** – strong, drum-like percussive accents
- **Sample 3** – A relatively regular, slightly melodic rhythm fragment

**In the course we already know that "loop = rhythm generation".** 
According to Margulis’ research on repetition, even irregular sequences become musical when looped extensively. This idea directly informed the design of my step-sequencer structure.

In `setup()`, audio samples were loaded via librosa:

```python
self.s1, _ = librosa.load("week3_week4_project/audio/sample_1.WAV", sr=self.sr)
self.s2, _ = librosa.load("week3_week4_project/audio/sample_2.WAV", sr=self.sr)
self.s3, _ = librosa.load("week3_week4_project/audio/sample_3.WAV", sr=self.sr)
```

A list-based sequence defined the temporal structure:
```python
self.sequence = [
    {"sound": "s1"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
    {"sound": "s3"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
    {"sound": "s3"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
    {"sound": "s1"}, {"sound": "s2"}, {"sound": "s3"}, {"sound": "s2"},
]
```

This list effectively functions as a **data structure encoding rhythm**—a key algorithmic element.
Each step calculates the starting point of the sample based on the beat position, which is then composited into a **NumPy** array and played back with **dot.music.startsamplestream()**.

---

# 3.Loop

In the early tests, I only played the sequence once. As a result, the music was too short, and it ended before it even started to enter the groove. After the loop, the rhythm gradually emerges, which is more in line with the repetitive characteristics of the original song.

**The length of the timeline and the sample are truncated**

At first, I wrote it in the lesson example: use a fixed length **blank timeline** to place the sample.

But the problem arises:

The blank is shorter than some samples, causing the sample to be forcibly truncated with a blank space in the middle.

It sounds very unnatural and also breaks the rhythm.

---

# 4.The final solution is:
Instead of setting the blank length in advance, the total length is dynamically calculated based on the end time of all placed samples.

```
    total_samples = total_steps * self.beat_samples
else:
    total_samples = max(end for (start, end, src) in events_with_sound)
```

This ensured full playback for all samples and preserved rhythmic continuity.

---

# 5.Visual Design: Recreating Northeastern Floral Aesthetics

Floral patterns on **Northeast China's floral cotton-padded jackets.**
For the visual design, I referenced the **floral cotton-padded jacket**, a style very characteristic of Northeast China. 

**Color scheme:**
- Earth red
- Bright green
- Vulgar yellow

The overall goal is to achieve an effect that is "vulgar yet powerful".

The `draw_flower()` function renders:

- layered green leaves
- red/pink petals
- a yellow centre
The background is a deep brown-red:
```
dot.background((120, 0, 0))
```
![image](https://git.arts.ac.uk/user-attachments/assets/59e917ab-0cce-4af5-b9de-c05158d6064d)


And many flowers are arranged around the perimeter of the image, creating a **crowded, over-decorated** effect—which is also the aesthetic characteristic of Northeast China that I want to emphasize.

---

# 6.Audio-Driven Animation
My first idea—scaling the flower size by amplitude—felt too smooth for the aggressive rhythm. I instead implemented **shake-based motion**, which better matches:

- roughness

- directness

- instability

Goes well with the drum beat/rhythm

But my first implementation failed because I triggered the shake with a "sample start/end event", resulting in irregular jitter that didn't reflect the musical structure.




https://github.com/user-attachments/assets/419e49f3-93eb-40a3-9de6-ba0aa9adecd7




Switching to continuous amplitude **energy** produced far more coherent motion.
```
amp = dot.music.amplitude()
amp_clamped = max(0.0, min(1.0, float(amp)))
```


Re-mapped to jitter amplitude:

```
mag = int(TEMP_SCALE * (amp_clamped - TEMP_THRESHOLD) / (1.0 - TEMP_THRESHOLD)) + 2
```


The visuals immediately became rhythmic.

But during debugging, I found that sample2 and sample3 barely shake when playing.
The reason is that their peak amplitude is much lower than Sample1.
**I adjusted the peaks of sample2 and sample3 to be consistent with sample1.**

```
max1 = np.max(np.abs(self.s1))
max2 = np.max(np.abs(self.s2))
max3 = np.max(np.abs(self.s3))

self.s2 = self.s2 * (max1 / (max2 + 1e-6))
self.s3 = self.s3 * (max1 / (max3 + 1e-6))
```

It is only when balanced that the performance of the shake truly reflects the structure of the music.

This made me realize that when doing audio-driven vision, **you need to not only **read the amplitude**, but also **equalize the input**, otherwise the visual mapping will be inaccurate. **








---

# 7.Summary and Reflection

**This project taught me several important principles relevant to digital media computation:
-1. Repetition can combine fragmented sounds into new rhythmic structures; 
-2. Real-world audio is not "clean", and problems such as sample length, uneven energy, and volume differences must be dealt with. 
-3. The visual mapping method should be chosen correctly, and the visual rhythm method suitable for different rhythms is different; 
-4. "Ugly" is also a design language, as long as it reflects cultural characteristics, it can be very appealing; 
-5. The mapping of audio and vision is not only a technique, but also an aesthetic judgment.**

Overall, this project helped me gain a **deeper understanding:**

How to organize audio time structures with `sequences`

How to manipulate and synthesize multi-segment samples with `NumPy`

How to **select and debug** audio-driven visual parameters

It also deepened my interest in the recreation of "earthy culture" in digital media。

# File Structure

```python
 main.py / project1.py — Main program

 audio/sample_1.WAV — Strong drum beat clip

 audio/sample_2.WAV — Funny transition

 audio/sample_3.WAV — Regular rhythm segments

 README.md — This document

```

# Operating mode

**Installation dependencies:**

```python
pip install numpy librosa opencv-python
```

**Make sure the directory is structured correctly**

```python
week3_week4_project/
    audio/
        sample_1.WAV
        sample_2.WAV
        sample_3.WAV
    project_1.py
    README_1.md
```


**Run**：

`project_1.py`

Dorothy opens the window, plays reconstruction music, and sees a Northeast floral padded jacket-style floral pattern that shakes with the rhythm.
