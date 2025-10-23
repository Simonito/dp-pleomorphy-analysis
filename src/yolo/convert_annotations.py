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
