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


## Workflow

1. **Data Preparation**: Raw slide data is stored in `data/raw/slides/`
2. **Segmentation**: Run notebooks in `notebooks/segmentation/` to perform segmentation
3. **Visualization**: Use notebooks in `notebooks/visualization/` to visualize results
4. **YOLO Conversion**: Convert annotations to YOLO format using notebooks in `notebooks/yolo_conversion/`

## Dependencies

See `pyproject.toml` for project dependencies.