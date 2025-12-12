# Week 6 – Manhole cover world map Photomosaic (project_2.py)

**Project Overview**

This week’s image-processing session introduced the idea that an image is fundamentally a *spatial distribution of light*, discretised into a matrix of pixel values. Alongside this technical perspective, my reading of **Crawford & Paglen’s Excavating AI** prompted me to consider images not only as numerical data but also as political artefacts shaped by decisions about collection, classification, and visibility.

In response, I created a photomosaic of a world map composed entirely of **manhole cover images**, a dataset typically ignored in mainstream visual culture. The project attempts to highlight how overlooked urban fragments can be recontextualised to construct a macro-level geopolitical symbol.

In this context, I made a work that seems simple, but in fact hides many choices and positions:  
> Piece together a world map from the manhole cover image dataset.

---

## 1. Concept: Reconstructing the World from Overlooked Urban Fragments

In this week's project, I used the manhole dataset to generate a photomosaic of a world map.

The manhole cover dataset I used:
<https://universe.roboflow.com/manhole-dataset/manhole-cover-1>
<img width="914" alt="dataset" src="https://git.arts.ac.uk/user-attachments/assets/4b62c658-f016-4309-8665-543e617993c2" />

**Manhole covers** embody several contradictions:
- They are essential infrastructure yet largely unnoticed. 
- They express **regional identity** while also being standardised through urbanisation. 
- They record traces of daily human life but usually remain in the visual background.

In contrast, the **world map** represents:
![22543a5O-2](https://git.arts.ac.uk/user-attachments/assets/d8c0989e-17d5-46d1-85e6-65f8016840f3)

- national borders, geopolitics, and territorial imaginaries
- a macro perspective often associated with power and abstraction

By constructing the map from manhole covers, I aimed to juxtapose:
- **micro ↔ macro**
- **mundane objects ↔ symbolic geography**
- **invisible fragments ↔ dominant representations**

I hope to create a stark contrast through this work: 
**Using manhole covers to construct the entire world.**
Land areas are composed of manhole covers, whereas oceans are rendered in blue tones for legibility.

---

## 2. From Pixels to Manhole Covers: Ultra-Large-Pixel Photomosaic
**2.1 Downsampling the World Map**
To obtain a grid where each block can be replaced by an image tile, I downsampled the world map:
```python
target_im_np = np.array(Image.open(src_image_path))
mosaic_template = np.swapaxes(
    target_im_np[::downsample_rate, ::downsample_rate], 0, 1
)
```
Reducing by `downsample_rate = 10` yields a template matrix where each value represents a “super-pixel” to be matched with a manhole cover.

**2.2 Computing the Average Color of Each Manhole Image**
```python
dataset = dot.get_images(dataset_path, thumbnail_size=thumbnail_size)
image_values = np.apply_over_axes(np.mean, dataset, [1, 2]).reshape(dataset.shape[0], 3)
tree = spatial.KDTree(image_values)
```
Each manhole cover image is converted into a thumbnail (64x64) and an RGB average vector (for color matching).
<img width="566" alt="image_1" src="https://git.arts.ac.uk/user-attachments/assets/c581bf11-cf73-4f7b-9c2c-5fe7b8725b27" />

For the full image, please open mosaic_1.png

---

## 3.Ocean Detection and Blue Correction
To maintain recognisability of the world map’s structure, I implemented ocean detection and handled ocean pixels separately.

**3.1 Identifying Ocean Pixels**

```python
def is_ocean_color(rgb):
    r, g, b = rgb
    return (b > r + 30) and (b > g + 30) and (b > 100)
```
A pixel is labelled “ocean” if:
- Blue dominates both red and green by a defined threshold
- Blue intensity is sufficiently high

**3.2 Blue-Tone Correction for Ocean Areas**
  
Instead of placing manhole covers in ocean regions, I apply a blue correction:
  
```python
  def adjust_to_blue_tone(rgb):
    r, g, b = rgb
    blue_factor = 1.2
    new_r = int(r * 0.3)
    new_g = int(g * 0.5)
    new_b = int(min(255, b * blue_factor))
    return (new_r, new_g, new_b)
```

By **correcting** the blue tones, I made the ocean area more uniform and clean, avoiding the distraction of manhole cover images. 
This is also a **visual and political choice**:

> I decided who could become the building blocks of the "world" (land - manhole cover), and who could only exist as background (ocean - blue block).

This improves clarity and foregrounds the landmass, reinforcing conceptual distinctions between:
- what is *selected* vs. what is ignored
- what becomes *visible* vs. what becomes background
<img width="971" alt="image_2" src="https://git.arts.ac.uk/user-attachments/assets/a803c30a-38bc-482c-8e24-99474f2752b3" />
For the full image, please open mosaic_2.png

---

## 4.k = 5：Balancing Similarity and Variability
```python
match = tree.query(template, k=k)
pick = np.random.randint(k)
image_idx[i, j] = match[1][pick]
```
Setting `k = 5` ensures:
- Only colour-similar manhole covers are considered.
- Randomised selection among neighbours increases visual richness.
- The matching process avoids repetitive uniformity.

---

## 5.Reflection：
This project, combining coding and reading, taught me that photomosaic is not just a visual effect, but a way of **deconstructing and reconstructing** images and data. 
The classification system carries my **subjectively assigned values** ​​(ocean and land, manhole cover images and blue blocks, visible and invisible parts). In my project, I'm not describing my world, but **constructing** it. 
In this work, I attempted to temporarily make those manhole covers, which are normally unnoticed and underfoot, the focus of the viewer.

**If the dataset of manhole covers I find is missing manhole covers from a certain region, then my method of using manhole covers to construct a "world" is somewhat biased.**

After completing my project, I deeply reflected on the question of whether the work took a neutral stance. 
**The manhole cover images, world map, sea route separation, and color matching used in my project were not neutral; they collectively constituted an "urban-centric" perspective.**
**My subjective idea of ​​using highly detailed manhole cover elements as pixel blocks to construct a "world" inherently carries my own values, therefore, my project does reflect my own stance.**

---

## Operating mode
**1.Install dependencies**
```python
pip install numpy pillow scipy dorothy-cci
```
Illustrate:
- `numpy` is used to process image matrices
- `Pillow (PIL)` is used to read and write images
- `scipy` provides KDTree for nearest neighbor search
- `dorothy-cci` for bulk loading thumbnail datasets (dot.get_images)
  
**2.Project Structure**
```python
Coding_for_Digital_Media_Project_Xin_Wang/
    week6_project/
        data/       # Since the database file is too big I put the link in this file
            test/    # <https://universe.roboflow.com/manhole-dataset/manhole-cover-1>
            train/
            val/
        image_world map/
            22543a5O-2.jpg       
        mosaic_1.png
        mosaic_2.png
        project_2.py
```
**3.Run**
```python
project_2.py
```
Program pipeline:

1.Read and downsample world map

2.Load dataset & compute average colours

3.Use KDTree for matching

4.Apply blue correction for ocean areas

5.Construct final mosaic → `mosaic.png`

After the build is complete, you can find it in the **week6_project/directory: mosaic.png**
