import numpy as np
from PIL import Image
from scipy import spatial
import PIL
from dorothy import Dorothy

dot = Dorothy()

# Source image to be converted into a photomosaic (world map).
src_image_path = r'C:\Users\wangxinxin\Documents\GitHub\Coding_for_Digital_Media_Project_Xin_Wang\week6_project\image_world map\22543a5O-2.jpg'
# Folder containing the thumbnail dataset used to build the mosaic.
dataset_path = r"C:\Users\wangxinxin\Documents\GitHub\Coding_for_Digital_Media_Project_Xin_Wang\week6_project\data\test"
# Folder containing the thumbnail dataset used to build the mosaic.
thumbnail_size = (64, 64)
downsample_rate = 10
# Number of nearest neighbours to consider in the KDTree.
k = 5

target_im_np = np.array(Image.open(src_image_path))
# Downsample and transpose to form a coarse 'mosaic template'.
mosaic_template = np.swapaxes(target_im_np[::downsample_rate, ::downsample_rate], 0, 1)

# Load dataset of images as a 4D array
dataset = dot.get_images(dataset_path, thumbnail_size=thumbnail_size)
# Compute mean RGB value for each image in the dataset.
image_values = np.apply_over_axes(np.mean, dataset, [1, 2]).reshape(dataset.shape[0], 3)

tree = spatial.KDTree(image_values)

target_res = mosaic_template.shape[0:2]
image_idx = np.zeros(target_res, dtype=np.uint32)

is_ocean = np.zeros(target_res, dtype=bool)
ocean_colors = np.zeros((*target_res, 3), dtype=np.uint8)

def is_ocean_color(rgb):
    r, g, b = rgb
    return (b > r + 30) and (b > g + 30) and (b > 100)


def adjust_to_blue_tone(rgb):
    r, g, b = rgb
    # Perceptual luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    blue_factor = 1.2
    new_r = int(r * 0.3)
    new_g = int(g * 0.5)
    new_b = int(min(255, b * blue_factor))

    return (new_r, new_g, new_b)

# - find the nearest thumbnail by mean colour using KDTree.
for i in range(target_res[0]):
    for j in range(target_res[1]):
        template = mosaic_template[i, j]

        if is_ocean_color(template):
            is_ocean[i, j] = True
            ocean_colors[i, j] = adjust_to_blue_tone(template)
        else:

            # Find k nearest thumbnails and randomly pick one of them.
            match = tree.query(template, k=k)
            if k == 1:
                image_idx[i, j] = match[1]
            else:
                pick = np.random.randint(k)
                image_idx[i, j] = match[1][pick]

# Create the blank mosaic image with full resolution.
mosaic = PIL.Image.new('RGB', (thumbnail_size[0] * target_res[0], thumbnail_size[1] * target_res[1]))

for i in range(target_res[0]):
    for j in range(target_res[1]):
        arr = dataset[image_idx[i, j]]
        x, y = i * thumbnail_size[0], j * thumbnail_size[1]

        if is_ocean[i, j]:
            # Use a solid blue-toned tile for ocean regions.
            blue_color = tuple(ocean_colors[i, j])
            ocean_tile = PIL.Image.new('RGB', thumbnail_size, blue_color)
            mosaic.paste(ocean_tile, (x, y))
        else:
            # Use matching thumbnail from dataset.
            arr = dataset[image_idx[i, j]]
            im = PIL.Image.fromarray(arr)
            mosaic.paste(im, (x, y))

mosaic.save('mosaic_2.png')