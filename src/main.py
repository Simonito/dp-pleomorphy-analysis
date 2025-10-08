import json
import os
import numpy as np
from large_image import getTileSource
import tifffile
import openslide
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


OUT_DIR = '../outputs'

def load_geojson(geojson_path):
    """Load geojson file containing annotations."""
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    return geojson_data

def find_bounding_rectangle(geojson_data):
    """
    Find the bounding rectangle that contains all annotations.
    Returns: (min_x, min_y, max_x, max_y) - the coordinates of the rectangle
    """
    all_points = []
    
    # Extract all points from all features
    for feature in geojson_data.get('features', []):
        geometry = feature.get('geometry', {})
        geometry_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geometry_type == 'Polygon':
            # For polygons, coordinates are [exterior_ring, interior_ring1, ...]
            for ring in coordinates:
                all_points.extend(ring)
        elif geometry_type == 'MultiPolygon':
            # For multipolygons, coordinates are [polygon1, polygon2, ...]
            for polygon in coordinates:
                for ring in polygon:
                    all_points.extend(ring)
        elif geometry_type == 'LineString':
            all_points.extend(coordinates)
        elif geometry_type == 'Point':
            all_points.append(coordinates)
    
    if not all_points:
        raise ValueError("No valid geometries found in the geojson file")
    
    # Convert to numpy array for easier operations
    points = np.array(all_points)
    
    # Get min and max coordinates
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    return min_x, min_y, max_x, max_y

def remap_annotations(geojson_data, offset_x, offset_y):
    """
    Remap annotations coordinates by subtracting the offset.
    This shifts all annotations to be relative to the top-left of the extracted region.
    """
    remapped_geojson = geojson_data.copy()
    
    for feature in remapped_geojson.get('features', []):
        geometry = feature.get('geometry', {})
        geometry_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geometry_type == 'Polygon':
            for i, ring in enumerate(coordinates):
                remapped_ring = []
                for point in ring:
                    remapped_ring.append([point[0] - offset_x, point[1] - offset_y])
                coordinates[i] = remapped_ring
        
        elif geometry_type == 'MultiPolygon':
            for i, polygon in enumerate(coordinates):
                remapped_polygon = []
                for ring in polygon:
                    remapped_ring = []
                    for point in ring:
                        remapped_ring.append([point[0] - offset_x, point[1] - offset_y])
                    remapped_polygon.append(remapped_ring)
                coordinates[i] = remapped_polygon
        
        elif geometry_type == 'LineString':
            remapped_line = []
            for point in coordinates:
                remapped_line.append([point[0] - offset_x, point[1] - offset_y])
            feature['geometry']['coordinates'] = remapped_line
        
        elif geometry_type == 'Point':
            feature['geometry']['coordinates'] = [
                coordinates[0] - offset_x,
                coordinates[1] - offset_y
            ]
    
    return remapped_geojson

def extract_region_from_wsi(wsi_path, output_path, x_min, y_min, width, height, level=0):
    """
    Extract a region from the WSI image and save it using large_image.
    
    Args:
        wsi_path: Path to the WSI file (.mrxs format)
        output_path: Path to save the extracted region (as .ome.tif)
        x_min, y_min: Top-left coordinates of the region to extract
        width, height: Width and height of the region to extract
        level: Not used with large_image, but kept for compatibility
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Open the WSI file with large_image
    source = getTileSource(wsi_path)
    
    # Get metadata for resolution information
    metadata = source.getMetadata()
    mpp_x = float(metadata.get("mm_x", 0)) * 1000  # convert mm to microns
    mpp_y = float(metadata.get("mm_y", 0)) * 1000
    
    print(f"WSI Resolution: {mpp_x} × {mpp_y} microns per pixel")
    
    # Define the region to extract
    region = {
        'left': int(x_min),
        'top': int(y_min),
        'width': int(width),
        'height': int(height),
        'units': 'base_pixels'
    }
    
    print(f"Extracting region: {region}")
    
    # Get the region as a PIL image
    tile_image, _ = source.getRegion(region=region, format='PIL')
    
    # Convert to RGB numpy array
    tile_rgb = np.array(tile_image.convert("RGB"))
    
    # Save as OME-TIFF with resolution and compression
    tifffile.imwrite(
        output_path,
        tile_rgb,
        photometric='rgb',
        tile=(256, 256),  # tiled like WSIs
        compression='deflate',
        resolution=(1 / mpp_x, 1 / mpp_y) if mpp_x > 0 and mpp_y > 0 else None,
        resolutionunit='CENTIMETER',
        metadata={'axes': 'YXS'},
        ome=True
    )
    
    print(f"Extracted region saved to {output_path}")
    return output_path


def visualize_extraction(geojson_data, remapped_geojson, bounds):
    """Visualize the original annotations and the remapped annotations"""
    min_x, min_y, max_x, max_y = bounds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot original annotations
    ax1.set_title("Original Annotations")
    for feature in geojson_data.get('features', []):
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Polygon':
            polygon = Polygon(geometry.get('coordinates')[0])
            x, y = polygon.exterior.xy
            ax1.plot(x, y, 'r-')
    
    # Add the extraction rectangle
    rect_x = [min_x, max_x, max_x, min_x, min_x]
    rect_y = [min_y, min_y, max_y, max_y, min_y]
    ax1.plot(rect_x, rect_y, 'b--', linewidth=2)
    
    # Set limits to see the entire context
    margin = max((max_x - min_x), (max_y - min_y)) * 0.1
    ax1.set_xlim(min_x - margin, max_x + margin)
    ax1.set_ylim(min_y - margin, max_y + margin)
    
    # Plot remapped annotations
    ax2.set_title("Remapped Annotations")
    for feature in remapped_geojson.get('features', []):
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Polygon':
            polygon = Polygon(geometry.get('coordinates')[0])
            x, y = polygon.exterior.xy
            ax2.plot(x, y, 'r-')
    
    # Set limits to see just the extraction area
    width = max_x - min_x
    height = max_y - min_y
    ax2.set_xlim(-width * 0.05, width * 1.05)
    ax2.set_ylim(-height * 0.05, height * 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "visualization/annotation_visualization.png"))
    print("Visualization saved as annotation_visualization.png")
    plt.close()


GEOJSON_PATH = './data/raw/slide-2024-04-03T07-52-35-R1-S2.geojson'
geojson_data = load_geojson(GEOJSON_PATH)

min_x, min_y, max_x, max_y = find_bounding_rectangle(geojson_data)
width = max_x - min_x
height = max_y - min_y

print(f"min_x {min_x} | min_y {min_y}")
print(f"width {width} | height: {height}")

remapped_geojson = remap_annotations(geojson_data, min_x, min_y)

REMAPPED_GEOJSON = f'remapped_{os.path.basename(GEOJSON_PATH)}'
with open(os.path.join(OUT_DIR, REMAPPED_GEOJSON), 'w') as f:
    json.dump(remapped_geojson, f, indent=2)

visualize_extraction(geojson_data, remapped_geojson, (min_x, min_y, max_x, max_y))

INPUT_WSI_PATH = './data/raw/slide-2024-04-03T07-52-35-R1-S2.mrxs'
OUTPUT_WSI_PATH = './outputs/remapped_slide-2024-04-03T07-52-35-R1-S2.ome.tif'

extract_region_from_wsi(INPUT_WSI_PATH, OUTPUT_WSI_PATH, min_x, min_y, width, height, level=0)

#--------------- INSTANCE INFERENCE ------------------#

INFERENCE_INPUT_DATA_DIR = os.path.join(OUT_DIR, 'segmentation', 'inference-inputs')

def prepare_inputs():
    os.makedirs(INFERENCE_INPUT_DATA_DIR, exist_ok=True)
    shutil.copy(OUTPUT_WSI_PATH, os.path.join(INFERENCE_INPUT_DATA_DIR, os.path.basename(OUTPUT_WSI_PATH)))


prepare_inputs()
os.listdir(INFERENCE_INPUT_DATA_DIR)

import os
import pandas as pd
from tqdm.auto import tqdm
import torch
from pathlib import Path
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-i_p", "--image_path", type=str, default=INFERENCE_INPUT_DATA_DIR)
parser.add_argument("-m_f", "--model_folder", type=str, default="brightfield_nuclei")
parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
parser.add_argument("-exclude", "--exclude_str", type=str, default= ["mask","prediction", "geojson", "zip", "._"], help="Exclude files with this string in their name")
parser.add_argument("-pixel_size", "--pixel_size", type=float, default= None, help="Pixel size of the input image in microns")
parser.add_argument("-recursive", "--recursive",default=False, type=lambda x: (str(x).lower() == 'true'),help="Look for images recursively at the image path")
parser.add_argument("-ignore_segmented", "--ignore_segmented",default=False, type=lambda x: (str(x).lower() == 'true'),help="Whether to ignore previously segmented images in the image path")

#advanced usage
parser.add_argument("-tile_size", "--tile_size", type=int, default= 512, help="tile size in pixels given to the model, only used for large images.")
parser.add_argument("-batch_size", "--batch_size", type=int, default= 3, help="batch size, only useful for large images")
parser.add_argument("-save_geojson", "--save_geojson", type=lambda x: (str(x).lower() == 'true'), default= True, help="Output geojson files of the segmentation")
parser.add_argument("-image_reader", "--image_reader", type=str, default= "tiffslide", help='The image reader to use. Options are "tiffslide", "skimage.io", "bioio", "AICSImageIO""')
parser.add_argument("-use_otsu", "--use_otsu_threshold", type=lambda x: (str(x).lower() == 'true'), default= True, help="Use an Otsu Threshold on the WSI thumbnail to determine which channels to segment(ignored for images that are not WSIs)")

parser.add_argument("-kwargs", "--kwargs", nargs="*", type=str, default={}, help="Additional keyword arguments in the form key=value", dest="kwargs_raw")

def smart_cast(value):
    """Try to convert string to int, float, bool, or leave as string."""
    value_lower = value.lower()
    if value[0] == "[" and value[-1] == "]":
        value = value.replace("[","").replace("]","").split(",")
        value = [smart_cast(v) for v in value]
        return value
    if value_lower == "true":
        return True
    elif value_lower == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def parse_key_value(arg_list):
    kwargs = {}
    for arg in arg_list:
        if "=" not in arg:
            raise argparse.ArgumentTypeError(f"Invalid format for argument '{arg}'. Use key=value.")
        key, value = arg.split("=", 1)
        kwargs[key] = smart_cast(value)
    return kwargs


def file_matches_requirement(root,file, exclude_str):
    if not os.path.isfile(os.path.join(root,file)):
        return False
    for e_str in exclude_str:
        if e_str in file:
            return False
        if parser.ignore_segmented:
            for extension in [".tiff",".zarr"]:
                if os.path.exists(os.path.join(root,str(Path(file).stem) + prediction_tag + extension)):
                    return False
    return True

prediction_tag = "_instanseg_prediction"

parser, _ = parser.parse_known_args()

from instanseg import InstanSeg


# override the env variable which they take as the model location
os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = os.path.abspath("../data/model/") + "/"



# Convert the list of key=value strings into a dictionary
kwargs = parse_key_value(parser.kwargs_raw)
del parser.kwargs_raw


if parser.image_path is None or not os.path.exists(parser.image_path):
    print("image path is NAN or not exists")

if parser.model_folder is None:
    raise ValueError("Please provide a model name")

# TEST
import os
print("Path check:", os.path.exists(parser.model_folder))
# TEST

instanseg = InstanSeg(model_type=parser.model_folder, device=parser.device, image_reader=parser.image_reader)
instanseg.prediction_tag = prediction_tag

if not parser.recursive:
    print("Loading files from: ", parser.image_path)
    files = os.listdir(parser.image_path)
    files = [os.path.join(parser.image_path, file) for file in files if file_matches_requirement(parser.image_path, file, parser.exclude_str)]
else:
    print("Loading files recursively from: ", parser.image_path)
    files = []
    for root, dirs, filenames in os.walk(parser.image_path):
        for filename in filenames:
            if file_matches_requirement(root , filename, parser.exclude_str):
                files.append(os.path.join(root, filename))

assert len(files) > 0, "No files found in the specified directory"


for idx, file in enumerate(tqdm(files)):
    if idx == 4:
        break
    print("Processing: ", file)

    #breakpoint()

    _ = instanseg.eval(image=file,
                    pixel_size = parser.pixel_size,
                    save_output = True,
                    save_overlay = True,
                    save_geojson = parser.save_geojson,
                    batch_size = parser.batch_size,
                    tile_size = parser.tile_size,
                    use_otsu_threshold = parser.use_otsu_threshold,
                    **kwargs,
                    )
