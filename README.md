# SAR-ShipDet-Toolkit

A unified tool for processing various SAR (Synthetic Aperture Radar) ship detection datasets into a standardized format. This tool supports multiple popular datasets including HRSID, SAR-Ship-Dataset, SSDD, and SRSDD-V1.0, with robust error handling and comprehensive logging.

## Features

- Unified processing pipeline for multiple SAR ship detection datasets
- Support for both relative and absolute coordinate formats
- Automatic image padding to target dimensions (default 1024x1024)
- Optional visualization of bounding box annotations with customizable colors and thickness
- Comprehensive error handling and detailed logging
- Reproducible random padding with configurable seed
- Support for multi-class and single-class annotations
- Configurable output formats and directories
- Robust coordinate validation and boundary checking
- Automatic conversion between different bounding box formats (YOLO, COCO, DOTA)

## Supported Datasets

1. **HRSID**: High-Resolution SAR Images Dataset
   - Single class (ship)
   - COCO format annotations (.json)
   - Variable image sizes
   - Conversion: COCO format → DOTA format → Relative coordinates

2. **SAR-Ship-Dataset**
   - Single class (ship)
   - YOLO format annotations (.txt)
   - Fixed 256x256 images
   - Conversion: YOLO format → DOTA format → Relative coordinates

3. **SSDD**: SAR Ship Detection Dataset
   - Single class (ship)
   - XML annotations with rotated bounding boxes
   - Variable image sizes
   - Conversion: XML format → DOTA format → Relative coordinates

4. **SRSDD-V1.0**: SAR Rotation Ship Detection Dataset
   - Multi-class support (7 ship types)
   - Custom TXT format annotations with absolute coordinates
   - Fixed 1024x1024 images
   - Class mapping:
     ```python
     {
         "ore-oil": 0,
         "bulk-cargo": 1,
         "Fishing": 2,
         "LawEnforce": 3,
         "Dredger": 4,
         "Container": 5,
         "Cell-Container": 6
     }
     ```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- Python 3.7+
- PIL (Pillow)
- numpy
- logging

### Project Structure

```
├── utils.py                    # Core utility functions for coordinate conversion and image processing
├── UnifiedDatasetProcessor.py  # Main dataset processing class
├── process_sar_dataset.py      # Example processing script
├── requirements.txt
└── outputs/
    ├── split/ 
        ├── images/                 # Processed images
        ├── labels/                 # Processed labels
        └── visualizations/         # Optional visualization outputs
```

## Usage

### Basic Usage with Processing Config

```python
from UnifiedDatasetProcessor import DatasetProcessor, ProcessingConfig, CoordinateFormat

# Configure the processor
config = ProcessingConfig(
    input_dir="/path/to/dataset",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True  # Enable visualization
)

# Initialize and run processor
processor = DatasetProcessor(config)

# Process specific datasets
processor.process_hrsid()      # For HRSID
processor.process_sar_ship()   # For SAR-Ship-Dataset
processor.process_ssdd()       # For SSDD
processor.process_srsdd()      # For SRSDD-V1.0
```

### Advanced Configuration Options

```python
@dataclass
class ProcessingConfig:
    input_dir: str                    # Input dataset directory
    output_dir: str                   # Output directory for processed files
    target_size: Tuple[int, int]      # Target dimensions (width, height)
    coord_format: CoordinateFormat    # RELATIVE or ABSOLUTE
    visualize: bool = False           # Enable visualization
    visualization_dir: Optional[str] = None  # Custom visualization directory
```

### Coordinate Format Options

1. **Relative Format** (default)
   - All coordinates normalized to [0, 1]
   - Format: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
   - Example: `0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4`

2. **Absolute Format**
   - Coordinates in pixels
   - Format: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
   - Example: `0 100 200 300 200 300 400 100 400`

## Input Dataset Requirements

### HRSID
```
HRSID_PNG/
├── annotations/
│   ├── train2017.json
│   └── test2017.json
└── images/
    ├── train/
    └── test/
```

### SAR-Ship-Dataset
```
SAR-Ship-Dataset/
├── [image_name].jpg
└── [image_name].txt  # Matching label file
```

### SSDD
```
RBox_SSDD/voc_style/
├── JPEGImages_train/
├── JPEGImages_test/
├── Annotations_train/
└── Annotations_test/
```

### SRSDD-V1.0
```
SRSDD-V1.0/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Key Features Implementation

### Error Handling and Logging
- Comprehensive error catching and reporting
- Detailed logging with timestamps
- Graceful handling of invalid coordinates
- Warnings for potential issues

### Image Processing
```python
# Resize and pad images
processed_img, processed_labels = process_image_and_labels(
    image, labels, target_width, target_height
)

# Random padding with seed for reproducibility
padded_img, padded_labels = random_pad_image_and_rel_labels(
    image, labels, target_width, target_height, random_seed=42
)
```

### Coordinate Conversion
```python
# Convert between relative and absolute coordinates
rel_coords = convert_absolute_to_relative(abs_coords, width, height)
abs_coords = convert_relative_to_absolute(rel_coords, width, height)

# Convert between bbox formats
dota_bbox = convert_yolo_to_dota(yolo_bbox)
dota_bbox = convert_coco_to_dota(coco_bbox)
```

### Visualization
```python
# Visualize with custom settings
visualized_img = visualize_labels(
    image,
    labels,
    color=(255, 0, 0),  # Red boxes
    thickness=2
)
```

## Output Format

All datasets are converted to a unified format:
```
output_dir/
    ├── split/
        ├── images/
        │   └── {dataset_name}_{original_filename}.png
        ├── labels/
        │   └── {dataset_name}_{original_filename}.txt
        └── visualizations/
            └── {dataset_name}_{original_filename}_vis.png
```

## Citation

If you use this processor in your research, please cite:

```bibtex
@software{sar_ship_dataset_processor,
  title={SAR Ship Dataset Processor},
  author={SeonHoon Kim},
  year={2024},
  description={A unified tool for processing various SAR ship detection datasets}
}
```
