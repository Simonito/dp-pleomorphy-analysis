import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
import ast
from typing import Callable, Dict, List, Tuple
from pathlib import Path
import tifffile
from large_image import getTileSource


class YOLOAnnotationConverter:
    """Convert remapped annotations to YOLO format for object detection with tiling support."""

    def __init__(
        self,
        remapped_geojson_path: str,
        extracted_region_path: str,
        output_dir: str,
        default_margin: int = 20,
        tile_size: int = 640
    ):
        self.remapped_geojson_path = remapped_geojson_path
        self.extracted_region_path = extracted_region_path
        self.output_dir = output_dir
        self.default_margin = default_margin
        self.tile_size = tile_size

        os.makedirs(output_dir, exist_ok=True)
        self.remapped_geojson = self._load_geojson(remapped_geojson_path)
        self.tile_source = getTileSource(extracted_region_path)
        meta = self.tile_source.getMetadata()
        self.img_width, self.img_height = meta.get("sizeX", 0), meta.get("sizeY", 0)
        # self.img_width, self.img_height = self._get_image_dimensions(extracted_region_path)
        self.gdf = self._convert_to_geodataframe(self.remapped_geojson)
        self.class_mapping = self._create_class_mapping()

    def _load_geojson(self, geojson_path: str) -> dict:
        with open(geojson_path, 'r') as f:
            return json.load(f)

    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        try:
            with tifffile.TiffFile(image_path) as tif:
                return tif.pages[0].imagewidth, tif.pages[0].imagelength
        except Exception:
            source = getTileSource(image_path)
            meta = source.getMetadata()
            return meta.get("sizeX", 0), meta.get("sizeY", 0)

    def _convert_to_geodataframe(self, geojson_data: dict) -> gpd.GeoDataFrame:
        features = []
        for feature in geojson_data.get('features', []):
            geometry = feature.get('geometry', {})
            props = feature.get('properties', {})
            classification = props.get('classification', '{}')
            if isinstance(classification, str):
                try:
                    cdict = ast.literal_eval(classification)
                    cname = cdict.get('name', 'unknown')
                except Exception:
                    cname = 'unknown'
            elif isinstance(classification, dict):
                cname = classification.get('name', 'unknown')
            else:
                cname = 'unknown'
            geom = None
            if geometry.get('type') == 'Polygon':
                geom = Polygon(geometry['coordinates'][0])
            features.append({
                'geometry': geom,
                'objectType': props.get('objectType', ''),
                'classification': classification,
                'class_name': cname
            })
        return gpd.GeoDataFrame(features)

    def _create_class_mapping(self) -> Dict[str, int]:
        uniq = self.gdf['class_name'].unique()
        return {n: i for i, n in enumerate(uniq)}

    def _generate_tiles(self) -> List[Tuple[int, int, int, int, int, int]]:
        """Generate (tile_id, x0, y0, x1, y1) with overlap to cover entire image."""
        tile_size = self.tile_size
        tiles = []
        tid = 0
        for y0 in range(0, self.img_height, tile_size):
            y1 = min(y0 + tile_size, self.img_height)
            # if not exact fit, shift to include edge
            if y1 - y0 < tile_size and y0 > 0:
                y0 = max(0, y1 - tile_size)
            for x0 in range(0, self.img_width, tile_size):
                x1 = min(x0 + tile_size, self.img_width)
                if x1 - x0 < tile_size and x0 > 0:
                    x0 = max(0, x1 - tile_size)
                tiles.append((tid, x0, y0, x1, y1))
                tid += 1
        return tiles

    def _convert_objects_for_tile(self, tile_box: Tuple[int, int, int, int], margin_fn) -> List[str]:
        """Convert all objects intersecting the tile_box into YOLO labels normalized to tile size."""
        x0, y0, x1, y1 = tile_box
        tile_rect = box(x0, y0, x1, y1)
        tile_labels = []
        for _, row in self.gdf.iterrows():
            geom = row['geometry']
            if geom is None or not geom.intersects(tile_rect):
                continue
            inter = geom.intersection(tile_rect)
            if inter.is_empty:
                continue
            minx, miny, maxx, maxy = inter.bounds
            width = maxx - minx
            height = maxy - miny
            margin = margin_fn(width, height)
            minx = max(x0, minx - margin)
            miny = max(y0, miny - margin)
            maxx = min(x1, maxx + margin)
            maxy = min(y1, maxy + margin)
            # local coordinates within tile
            local_x_center = ((minx + maxx) / 2 - x0) / (x1 - x0)
            local_y_center = ((miny + maxy) / 2 - y0) / (y1 - y0)
            local_w = (maxx - minx) / (x1 - x0)
            local_h = (maxy - miny) / (y1 - y0)
            cls = self.class_mapping.get(row['class_name'], 0)
            tile_labels.append(f"{cls} {local_x_center:.6f} {local_y_center:.6f} {local_w:.6f} {local_h:.6f}")
        return tile_labels
    
    def _extract_rgb_region_for_tile(self, tile_box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, float]:
        x0, y0, x1, y1 = tile_box
        w = x1 - x0
        h = y1 - y0

        region = {
            'left': int(x0),
            'top': int(y0),
            'width': int(w),
            'height': int(h),
            'units': 'base_pixels'
        }

        tile_image, _ = self.tile_source.getRegion(region=region, format='PIL')
        tile_rgb = np.array(tile_image.convert("RGB"))

        metadata = self.tile_source.getMetadata()
        mpp_x = float(metadata.get("mm_x", 0)) * 1000  # convert mm to microns
        mpp_y = float(metadata.get("mm_y", 0)) * 1000

        return tile_rgb, mpp_x, mpp_y

    def convert_tiled(self, adaptive: bool = True):
        """Convert annotations into YOLO format tiles."""
        margin_fn = (lambda w, h: 0.1 * max(w, h)) if adaptive else (lambda w, h: self.default_margin)
        tiles = self._generate_tiles()
        out_files = []
        for tid, x0, y0, x1, y1 in tiles:
            out_name = f"tile_{tid:04d}_{x0}_{y0}"

            tile_out_path = os.path.join(self.output_dir, f"{out_name}.tif")
            tile_rgb, mpp_x, mpp_y = self._extract_rgb_region_for_tile((x0, y0, x1, y1))
            tifffile.imwrite(
                tile_out_path,
                tile_rgb,
                photometric='rgb',
                tile=(256, 256),  # tiled like WSIs
                compression='deflate',
                resolution=(1 / mpp_x, 1 / mpp_y) if mpp_x > 0 and mpp_y > 0 else None,
                resolutionunit='CENTIMETER',
                metadata={'axes': 'YXS'},
                ome=True
            )

            tile_labels = self._convert_objects_for_tile((x0, y0, x1, y1), margin_fn)

            # do not save empty annotation
            # and do not append to out_files
            # -> we do save all the tiles though
            if not tile_labels:
                continue

            annot_out_path = os.path.join(self.output_dir, f"{out_name}.txt")
            with open(annot_out_path, "w") as f:
                f.write("\n".join(tile_labels))

            out_files.append({
                "annotation": annot_out_path,
                "tile": tile_out_path
            })
        return out_files

    # ---------- original saving logic retained ----------

    def save_class_mapping(self, output_name: str) -> str:
        path = os.path.join(self.output_dir, f"{output_name}_classes.json")
        with open(path, "w") as f:
            json.dump(self.class_mapping, f, indent=2)
        return path

    def process(self, output_prefix: str = "yolo_annotations", use_adaptive_margin: bool = True):
        """Main entrypoint for tiled YOLO export."""
        print("Generating YOLO tiles...")
        files = self.convert_tiled(adaptive=use_adaptive_margin)
        class_map = self.save_class_mapping(output_prefix)
        return {"tiles": files, "class_mapping": class_map}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert remapped annotations to YOLO format (tiled)")
    parser.add_argument("--remapped_geojson", type=str, required=True)
    parser.add_argument("--extracted_region", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../data/processed/yolo")
    parser.add_argument("--output_prefix", type=str, default="yolo_annotations")
    parser.add_argument("--default_margin", type=int, default=20)
    parser.add_argument("--tile_size", type=int, default=640)
    parser.add_argument("--use_adaptive_margin", action="store_true")
    args = parser.parse_args()

    converter = YOLOAnnotationConverter(
        args.remapped_geojson,
        args.extracted_region,
        args.output_dir,
        args.default_margin,
        args.tile_size,
    )
    result_paths = converter.process(args.output_prefix, args.use_adaptive_margin)
    print(f"Generated {len(result_paths['tiles'])} tiled annotation files.")
    print(f"Class mapping saved to: {result_paths['class_mapping']}")



# #!/usr/bin/env python3
# """
# Convert remapped annotations to YOLO format for object detection.
# This script takes the remapped annotations from the main.py script and converts them
# to YOLO-compatible bounding box annotations.
# """

# import os
# import json
# import numpy as np
# import geopandas as gpd
# from shapely.geometry import Polygon, box
# import ast
# from typing import Callable, Dict, List, Tuple, Union
# from pathlib import Path
# import tifffile
# from large_image import getTileSource


# class YOLOAnnotationConverter:
#     """Convert remapped annotations to YOLO format for object detection."""

#     def __init__(
#         self,
#         remapped_geojson_path: str,
#         extracted_region_path: str,
#         output_dir: str,
#         default_margin: int = 20,
#         tile_size: int = 640
#     ):
#         """
#         Initialize the converter.

#         Args:
#             remapped_geojson_path: Path to the remapped geojson file
#             extracted_region_path: Path to the extracted region image
#             output_dir: Directory to save the YOLO annotations
#             default_margin: Default margin to add around bounding boxes
#         """
#         self.remapped_geojson_path = remapped_geojson_path
#         self.extracted_region_path = extracted_region_path
#         self.output_dir = output_dir
#         self.default_margin = default_margin
#         self.tile_size = tile_size
        
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Load the remapped geojson
#         self.remapped_geojson = self._load_geojson(remapped_geojson_path)
        
#         # Get image dimensions
#         self.img_width, self.img_height = self._get_image_dimensions(extracted_region_path)
        
#         # Convert to GeoDataFrame for easier processing
#         self.gdf = self._convert_to_geodataframe(self.remapped_geojson)
        
#         # Extract class names and create mapping
#         self.class_mapping = self._create_class_mapping()

#     def _load_geojson(self, geojson_path: str) -> dict:
#         """Load geojson file."""
#         with open(geojson_path, 'r') as f:
#             return json.load(f)

#     def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
#         """Get image dimensions from the extracted region."""
#         try:
#             # Try using tifffile first
#             with tifffile.TiffFile(image_path) as tif:
#                 img_width = tif.pages[0].imagewidth
#                 img_height = tif.pages[0].imagelength
#         except Exception:
#             # Fallback to large_image
#             source = getTileSource(image_path)
#             metadata = source.getMetadata()
#             img_width = metadata.get("sizeX", 0)
#             img_height = metadata.get("sizeY", 0)
            
#         return img_width, img_height

#     def _convert_to_geodataframe(self, geojson_data: dict) -> gpd.GeoDataFrame:
#         """Convert geojson to GeoDataFrame."""
#         features = []
        
#         for feature in geojson_data.get('features', []):
#             geometry = feature.get('geometry', {})
#             properties = feature.get('properties', {})
            
#             # Extract classification if available
#             classification = properties.get('classification', '{}')
#             if isinstance(classification, str):
#                 try:
#                     classification_dict = ast.literal_eval(classification)
#                     class_name = classification_dict.get('name', 'unknown')
#                 except (SyntaxError, ValueError):
#                     class_name = 'unknown'
#             elif isinstance(classification, dict):
#                 class_name = classification.get('name', 'unknown')
#             else:
#                 class_name = 'unknown'
            
#             # Create a feature dictionary
#             feature_dict = {
#                 'geometry': Polygon(geometry['coordinates'][0]) if geometry['type'] == 'Polygon' else None,
#                 'objectType': properties.get('objectType', ''),
#                 'classification': classification,
#                 'class_name': class_name
#             }
            
#             features.append(feature_dict)
        
#         return gpd.GeoDataFrame(features)

#     def _create_class_mapping(self) -> Dict[str, int]:
#         """Create a mapping from class names to class indices."""
#         unique_classes = self.gdf['class_name'].unique()
#         return {name: i for i, name in enumerate(unique_classes)}

#     def convert_with_fixed_margin(self) -> Tuple[List[str], gpd.GeoDataFrame]:
#         """
#         Convert annotations to YOLO format with a fixed margin.
        
#         Returns:
#             Tuple containing:
#                 - List of YOLO annotation strings
#                 - GeoDataFrame with expanded bounding boxes
#         """
#         return self._convert_with_margin(lambda w, h: self.default_margin)

#     def convert_with_adaptive_margin(self) -> Tuple[List[str], gpd.GeoDataFrame]:
#         """
#         Convert annotations to YOLO format with an adaptive margin based on object size.
        
#         Returns:
#             Tuple containing:
#                 - List of YOLO annotation strings
#                 - GeoDataFrame with expanded bounding boxes
#         """
#         def adaptive_margin(width, height):
#             return 0.1 * max(width, height)
        
#         return self._convert_with_margin(adaptive_margin)

#     def _convert_with_margin(self, margin_fn: Callable[[int, int], int]) -> Tuple[List[str], gpd.GeoDataFrame]:
#         """
#         Convert annotations to YOLO format with the specified margin function.
        
#         Args:
#             margin_fn: Function that takes width and height and returns margin size
            
#         Returns:
#             Tuple containing:
#                 - List of YOLO annotation strings
#                 - GeoDataFrame with expanded bounding boxes
#         """
#         labels = []
#         features = []
        
#         for _, row in self.gdf.iterrows():
#             if row['geometry'] is None:
#                 continue
                
#             # Get class index
#             cls = self.class_mapping.get(row['class_name'], 0)
            
#             # Get bounding box
#             minx, miny, maxx, maxy = row['geometry'].bounds
            
#             # Calculate margin
#             width = maxx - minx
#             height = maxy - miny
#             margin = margin_fn(width, height)
            
#             # Apply margin
#             minx = max(0, minx - margin)
#             miny = max(0, miny - margin)
#             maxx = min(self.img_width, maxx + margin)
#             maxy = min(self.img_height, maxy + margin)
            
#             # Calculate YOLO format (x_center, y_center, width, height) - normalized [0,1]
#             x_center = (minx + maxx) / 2 / self.img_width
#             y_center = (miny + maxy) / 2 / self.img_height
#             w = (maxx - minx) / self.img_width
#             h = (maxy - miny) / self.img_height
            
#             # Create YOLO annotation string
#             label = f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
#             labels.append(label)
            
#             # Create expanded bounding box for visualization
#             x_center_abs = (minx + maxx) / 2
#             y_center_abs = (miny + maxy) / 2
#             w_abs = (maxx - minx)
#             h_abs = (maxy - miny)
            
#             rect = box(x_center_abs - w_abs/2, y_center_abs - h_abs/2, 
#                        x_center_abs + w_abs/2, y_center_abs + h_abs/2)
            
#             feature = {
#                 'objectType': row['objectType'],
#                 'classification': row['classification'],
#                 'class_name': row['class_name'],
#                 'geometry': rect
#             }
#             features.append(feature)
        
#         return labels, gpd.GeoDataFrame(features)

#     def save_yolo_annotations(self, labels: List[str], output_name: str) -> str:
#         """
#         Save YOLO annotations to a text file.
        
#         Args:
#             labels: List of YOLO annotation strings
#             output_name: Base name for the output file
            
#         Returns:
#             Path to the saved file
#         """
#         output_path = os.path.join(self.output_dir, f"{output_name}.txt")
#         with open(output_path, "w") as f:
#             f.write("\n".join(labels))
#         return output_path

#     def save_expanded_boxes(self, gdf: gpd.GeoDataFrame, output_name: str) -> str:
#         """
#         Save expanded bounding boxes as a GeoJSON file for visualization.
        
#         Args:
#             gdf: GeoDataFrame with expanded bounding boxes
#             output_name: Base name for the output file
            
#         Returns:
#             Path to the saved file
#         """
#         output_path = os.path.join(self.output_dir, f"{output_name}.geojson")
#         gdf.to_file(output_path, driver="GeoJSON")
#         return output_path

#     def save_class_mapping(self, output_name: str) -> str:
#         """
#         Save class mapping to a JSON file.
        
#         Args:
#             output_name: Base name for the output file
            
#         Returns:
#             Path to the saved file
#         """
#         output_path = os.path.join(self.output_dir, f"{output_name}_classes.json")
#         with open(output_path, "w") as f:
#             json.dump(self.class_mapping, f, indent=2)
#         return output_path

#     def process(self, output_prefix: str = "yolo_annotations", use_adaptive_margin: bool = True) -> Dict[str, str]:
#         """
#         Process the annotations and save the results.
        
#         Args:
#             output_prefix: Prefix for output files
#             use_adaptive_margin: Whether to use adaptive margin
            
#         Returns:
#             Dictionary with paths to the saved files
#         """
#         if use_adaptive_margin:
#             labels, expanded_gdf = self.convert_with_adaptive_margin()
#             output_name = f"{output_prefix}_adaptive"
#         else:
#             labels, expanded_gdf = self.convert_with_fixed_margin()
#             output_name = f"{output_prefix}_fixed"
        
#         # Save results
#         txt_path = self.save_yolo_annotations(labels, output_name)
#         geojson_path = self.save_expanded_boxes(expanded_gdf, f"expanded_{output_name}")
#         class_path = self.save_class_mapping(output_name)
        
#         return {
#             "annotations": txt_path,
#             "expanded_boxes": geojson_path,
#             "class_mapping": class_path
#         }


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Convert remapped annotations to YOLO format")
#     parser.add_argument("--remapped_geojson", type=str, required=True, 
#                         help="Path to the remapped geojson file")
#     parser.add_argument("--tile_size", type=int, default=640,
#                         help="The size of the individual tiles after extracting annotations area")
#     parser.add_argument("--extracted_region", type=str, required=True,
#                         help="Path to the extracted region image")
#     parser.add_argument("--output_dir", type=str, default="../../data/processed/yolo",
#                         help="Directory to save the YOLO annotations")
#     parser.add_argument("--output_prefix", type=str, default="yolo_annotations",
#                         help="Prefix for output files")
#     parser.add_argument("--default_margin", type=int, default=20,
#                         help="Default margin to add around bounding boxes")
#     parser.add_argument("--use_adaptive_margin", action="store_true",
#                         help="Use adaptive margin based on object size")
    
#     args = parser.parse_args()
    
#     converter = YOLOAnnotationConverter(
#         args.remapped_geojson,
#         args.extracted_region,
#         args.output_dir,
#         args.default_margin,
#         args.tile_size
#     )
    
#     result_paths = converter.process(args.output_prefix, args.use_adaptive_margin)
    
#     print(f"YOLO annotations saved to: {result_paths['annotations']}")
#     print(f"Expanded boxes saved to: {result_paths['expanded_boxes']}")
#     print(f"Class mapping saved to: {result_paths['class_mapping']}")