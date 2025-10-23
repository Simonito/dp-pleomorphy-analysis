from geojson import Feature, FeatureCollection, Polygon
import json

def yolo_to_geojson(label_file, image_width, image_height, output_file="bboxes.geojson"):
    features = []

    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Label file {label_file} not found.")
        return
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            class_id, center_x, center_y, width, height = map(float, line.strip().split())
        except ValueError:
            print(f"Warning: Invalid format in line {i+1}: {line.strip()}. Skipping.")
            continue

        center_x_px = center_x * image_width
        center_y_px = center_y * image_height
        width_px = width * image_width
        height_px = height * image_height

        x_min = center_x_px - (width_px / 2)
        x_max = center_x_px + (width_px / 2)
        y_min = center_y_px - (height_px / 2)
        y_max = center_y_px + (height_px / 2)
        
        # ensure coordinates are within image bounds
        x_min = max(0, min(x_min, image_width))
        x_max = max(0, min(x_max, image_width))
        y_min = max(0, min(y_min, image_height))
        y_max = max(0, min(y_max, image_height))
        
        # polygon (rectangle) for the bounding box
        # geojson uses [x, y] format
        coords = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]
        ]

        polygon = Polygon([coords])

        feature = Feature(
            geometry=polygon,
            properties={
                "id": i + 1,
                "class_id": int(class_id),
                "name": f"Object_{i+1}_Class_{int(class_id)}"
            }
        )
        features.append(feature)
    
    feature_collection = FeatureCollection(features)

    try:
        with open(output_file, "w") as f:
            json.dump(feature_collection, f, indent=2)
        print(f"GeoJSON file saved to {output_file}")
    except Exception as e:
        print(f"Error saving GeoJSON file: {e}")


def main():
    label_file = "/Users/simon/Documents/000_fiit/09_semester/DP/notebooks/pleomorphy-analysis/dp-pleomorphy-analysis/data/processed/yolo-initial-640/yolo_dataset/labels/train/tile_0008_1280_640.txt"
    image_width = 640
    image_height = 640
    yolo_to_geojson(label_file, image_width, image_height, output_file="/Users/simon/Downloads/yolo-tile_0008_1280_640.geojson")


if __name__ == "__main__":
    main()