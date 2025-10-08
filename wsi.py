import os
import numpy as np
from large_image import getTileSource
import tifffile


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

