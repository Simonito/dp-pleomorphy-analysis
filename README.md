# Pleomorphy Analysis Project

## Project Structure

This project has been organized into a structured directory layout to improve maintainability and clarity:

```
├── data/
│   ├── raw/                  # Raw data files
│   │   └── slides/           # Original slide data
│   ├── processed/            # Processed data files
│   │   ├── annotations/      # Annotation files
│   │   └── segmentation/     # Segmentation results
│   └── model/                # Model files
│       └── brightfield_nuclei/
├── notebooks/
│   ├── segmentation/         # Notebooks for segmentation tasks
│   ├── visualization/        # Notebooks for visualization tasks
│   └── yolo_conversion/      # Notebooks for YOLO format conversion
├── outputs/
│   ├── segmentation/         # Segmentation outputs
│   ├── visualization/        # Visualization outputs
│   └── yolo/                 # YOLO format outputs
└── src/                      # Source code
    ├── main.py               # Main script
    └── wsi.py                # Whole Slide Image utilities
```

## File Path Updates

When running notebooks, you'll need to update file paths to reflect the new directory structure. Here are the common path changes:

### In notebooks/yolo_conversion/

- Old: `./data/slide-2024-04-03T07-52-35-R1-S2.geojson`
- New: `../../data/raw/slides/slide-2024-04-03T07-52-35-R1-S2.geojson`

### In notebooks/visualization/

- Old: `./expanded_bboxes.geojson`
- New: `../../data/processed/annotations/expanded_bboxes.geojson`

- Old: `./data/slide-2024-04-03T07-52-35-R1-S2.mrxs`
- New: `../../data/raw/slides/slide-2024-04-03T07-52-35-R1-S2.mrxs`

### In notebooks/segmentation/

- Old: `./outputs/`
- New: `../../outputs/`

## Workflow

1. **Data Preparation**: Raw slide data is stored in `data/raw/slides/`
2. **Segmentation**: Run notebooks in `notebooks/segmentation/` to perform segmentation
3. **Visualization**: Use notebooks in `notebooks/visualization/` to visualize results
4. **YOLO Conversion**: Convert annotations to YOLO format using notebooks in `notebooks/yolo_conversion/`

## Dependencies

See `pyproject.toml` for project dependencies.