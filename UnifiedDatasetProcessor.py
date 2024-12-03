import os
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import Enum
from PIL import Image
from utils import *

class SRSDDClasses(Enum):
    ORE_OIL = 0
    BULK_CARGO = 1
    FISHING = 2
    LAW_ENFORCE = 3
    DREDGER = 4
    CONTAINER = 5
    CELL_CONTAINER = 6

    @classmethod
    def get_id(cls, name: str) -> int:
        """Get class ID from name, accounting for different formats"""
        mapping = {
            "ore-oil": cls.ORE_OIL.value,
            "bulk-cargo": cls.BULK_CARGO.value,
            "Fishing": cls.FISHING.value,
            "LawEnforce": cls.LAW_ENFORCE.value,
            "Dredger": cls.DREDGER.value,
            "Container": cls.CONTAINER.value,
            "Cell-Container": cls.CELL_CONTAINER.value
        }
        return mapping.get(name, -1)

class CoordinateFormat(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

@dataclass
class ProcessingConfig:
    input_dir: str          # Input dataset directory
    output_dir: str         # Output directory for processed files
    target_size: Tuple[int, int]  # Target image dimensions (width, height)
    coord_format: CoordinateFormat  # RELATIVE or ABSOLUTE
    visualize: bool = False  # Enable visualization of annotations
    visualization_dir: Optional[str] = None  # Custom directory for visualizations

    def validate(self) -> None:
        """Validates configuration parameters"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        if self.target_size[0] <= 0 or self.target_size[1] <= 0:
            raise ValueError("Target dimensions must be positive")

class DatasetProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.config.validate()
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Creates necessary output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def process_srsdd(self) -> None:
        """
        Process SRSDD-V1.0 dataset.
        
        The dataset contains multi-class ship annotations in absolute coordinates.
        Each label file has two metadata lines followed by annotations in format:
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
        
        Raises:
            FileNotFoundError: If input directories don't exist
            ValueError: If image/label pairs are invalid
        """
        splits = ['train', 'test']
        
        for split in splits:
            # Set up paths
            input_image_dir = os.path.join(self.config.input_dir, split, 'images')
            input_label_dir = os.path.join(self.config.input_dir, split, 'labels')
            
            if not os.path.exists(input_image_dir) or not os.path.exists(input_label_dir):
                logging.error(f"SRSDD {split} directories not found")
                continue
            
            # Create split-specific output directories
            split_output_dir = os.path.join(self.config.output_dir, split)
            split_image_dir = os.path.join(split_output_dir, 'images')
            split_label_dir = os.path.join(split_output_dir, 'labels')
            os.makedirs(split_image_dir, exist_ok=True)
            os.makedirs(split_label_dir, exist_ok=True)
            
            # Process images
            for image_filename in os.listdir(input_image_dir):
                if not image_filename.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                try:
                    base_name = os.path.splitext(image_filename)[0]
                    image_path = os.path.join(input_image_dir, image_filename)
                    label_path = os.path.join(input_label_dir, base_name + '.txt')
                    
                    if not os.path.exists(label_path):
                        logging.warning(f"No label found for {image_filename}")
                        continue
                    
                    # Load image
                    image = Image.open(image_path)
                    original_width, original_height = image.size
                    
                    # Read labels
                    labels = self._read_srsdd_label(
                        label_path,
                        original_width,
                        original_height
                    )
                    
                    if not labels:
                        logging.warning(f"No valid labels found in {label_path}")
                        continue
                    
                    # Process image and labels
                    processed_img, processed_labels = process_image_and_labels(
                        image,
                        [label['coords'] for label in labels],
                        self.config.target_size[0],
                        self.config.target_size[1]
                    )
                    
                    # Convert coordinates if absolute format requested
                    if self.config.coord_format == CoordinateFormat.ABSOLUTE:
                        processed_labels = [
                            convert_relative_to_absolute(
                                label,
                                self.config.target_size[0],
                                self.config.target_size[1]
                            )
                            for label in processed_labels
                        ]
                    
                    # Save processed image
                    output_name = f"SRSDD_{base_name}"
                    processed_img.save(os.path.join(split_image_dir, f"{output_name}.png"))
                    
                    # Save labels with original class IDs
                    with open(os.path.join(split_label_dir, f"{output_name}.txt"), 'w') as f:
                        for idx, label_coords in enumerate(processed_labels):
                            class_id = labels[idx]['class_id']
                            label_str = f"{class_id} {' '.join(map(str, label_coords))}"
                            f.write(label_str + '\n')
                    
                    # Visualize if requested
                    if self.config.visualize:
                        vis_img = visualize_labels(
                            processed_img,
                            [[label['class_id']] + list(coords) 
                             for label, coords in zip(labels, processed_labels)]
                        )
                        vis_dir = os.path.join(split_output_dir, 'visualizations')
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_img.save(os.path.join(vis_dir, f"{output_name}_vis.png"))
                        
                except Exception as e:
                    logging.error(f"Error processing SRSDD image {image_filename}: {str(e)}")
                    continue
                else:
                    logging.info(f"Processed: {image_filename}")

    def _read_srsdd_label(
        self,
        label_path: str,
        img_width: int,
        img_height: int
    ) -> List[Dict[str, Any]]:
        """
        Read and parse SRSDD label file.
        
        Args:
            label_path: Path to label file
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            List of dictionaries containing class_id and relative coordinates
        """
        labels = []
        
        try:
            with open(label_path, 'r') as f:
                # Skip metadata lines
                f.readline()  # imagesource
                f.readline()  # gsd
                
                # Process label lines
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10:  # Need at least coordinates, class name, and difficulty
                        continue
                    
                    try:
                        # Parse coordinates and convert to relative
                        coords = list(map(float, parts[:8]))
                        class_name = parts[8]
                        class_id = SRSDDClasses.get_id(class_name)
                        
                        if class_id == -1:
                            logging.warning(f"Unknown class {class_name} in {label_path}")
                            continue
                            
                        rel_coords = convert_absolute_to_relative(coords, img_width, img_height)
                        
                        labels.append({
                            'class_id': class_id,
                            'coords': rel_coords,
                            'difficulty': int(parts[9])
                        })
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Invalid label format in {label_path}: {str(e)}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error reading label file {label_path}: {str(e)}")
            return []
            
        return labels

    def process_hrsid(self) -> None:
        """
        Process HRSID dataset from HRSID_PNG directory structure.
        
        Expected directory structure:
        HRSID_PNG/
        ├── annotations/
        │   ├── train2017.json
        │   └── test2017.json
        └── images/
            └── [image files]
        """
        logging.info("Starting HRSID dataset processing...")
        
        # Set up paths - using os.path.join with input_dir directly
        annotation_dir = os.path.join(self.config.input_dir, 'annotations')
        image_dir = os.path.join(self.config.input_dir, 'images')
        
        # Debug paths
        logging.info(f"Looking for annotations in: {annotation_dir}")
        logging.info(f"Looking for images in: {image_dir}")
        
        # Verify directories exist
        if not os.path.exists(annotation_dir):
            raise FileNotFoundError(f"Annotations directory not found: {annotation_dir}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
            
        # Process train and test splits
        splits = ['train2017', 'test2017']
        
        for split in splits:
            json_path = os.path.join(annotation_dir, f"{split}.json")
            if not os.path.exists(json_path):
                logging.warning(f"JSON file not found for {split}: {json_path}")
                continue
                
            logging.info(f"Processing {split} split...")
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                images = {img['id']: img for img in data['images']}
                annotations = data['annotations']
                
                # Create output directories for this split
                split_output_dir = os.path.join(self.config.output_dir, split.replace('2017', ''))
                split_image_dir = os.path.join(split_output_dir, 'images')
                split_label_dir = os.path.join(split_output_dir, 'labels')
                os.makedirs(split_image_dir, exist_ok=True)
                os.makedirs(split_label_dir, exist_ok=True)
                
                # Group annotations by image_id
                annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
                for ann in annotations:
                    image_id = ann['image_id']
                    if image_id not in annotations_by_image:
                        annotations_by_image[image_id] = []
                    annotations_by_image[image_id].append(ann)
                    
                # Process each image
                for image_id, ann_list in annotations_by_image.items():
                    try:
                        # Get the image info and update the paths
                        img_info = images[image_id].copy()
                        
                        # Create modified image info with split-specific directories
                        modified_img_info = {
                            'file_name': img_info['file_name'],
                            'output_dir': split_output_dir,  # Add split-specific output directory
                            'image_dir': image_dir,          # Add source image directory
                        }
                        
                        self._process_single_image(
                            modified_img_info,
                            ann_list,
                            dataset_name="HRSID",
                            convert_fn=convert_coco_to_dota,
                            output_dir=split_output_dir  # Pass split-specific output directory
                        )
                    except Exception as e:
                        logging.error(f"Error processing HRSID image {image_id}: {str(e)}")
                        continue
                        
            except FileNotFoundError:
                logging.error(f"HRSID annotation file not found: {json_path}")
                continue
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON file: {json_path}")
                continue
            except Exception as e:
                logging.error(f"Error processing {split} split: {str(e)}")
                continue
                
        logging.info("Completed HRSID dataset processing")

    def process_sar_ship(self) -> None:
        """Process SAR-Ship dataset from a single directory containing both images and labels"""
        image_extensions = {'.jpg', '.jpeg', '.png'}
        files_by_basename = {}

        # First, group files by their base names
        for filename in os.listdir(self.config.input_dir):
            base_name, ext = os.path.splitext(filename)
            if base_name not in files_by_basename:
                files_by_basename[base_name] = {'image': None, 'label': None}
                
            if ext.lower() in image_extensions:
                files_by_basename[base_name]['image'] = filename
            elif ext.lower() == '.txt':
                files_by_basename[base_name]['label'] = filename

        # Process each pair of files
        for base_name, files in files_by_basename.items():
            try:
                if not files['image'] or not files['label']:
                    logging.warning(f"Incomplete pair for {base_name}, skipping")
                    continue

                image_path = os.path.join(self.config.input_dir, files['image'])
                label_path = os.path.join(self.config.input_dir, files['label'])

                # Read and validate label file
                try:
                    with open(label_path, 'r') as f:
                        labels = [list(map(float, line.strip().split())) for line in f.readlines()]
                except Exception as e:
                    logging.error(f"Error reading label file {label_path}: {str(e)}")
                    continue

                # Process the image and labels
                self._process_single_image(
                    {'file_name': files['image']},
                    labels,
                    dataset_name="SARShip",
                    convert_fn=convert_yolo_to_dota
                )

            except Exception as e:
                logging.error(f"Error processing SAR-Ship pair {base_name}: {str(e)}")
                continue

    def process_ssdd(self) -> None:
        """
        Process SSDD dataset from VOC-style directory.
        Processes both train and test sets.
        
        Expected directory structure:
        input_dir/
        ├── Annotations_train/    # Train XML annotations
        ├── Annotations_test/     # Test XML annotations
        ├── JPEGImages_train/     # Train images
        └── JPEGImages_test/      # Test images
        """
        logging.info("Starting SSDD dataset processing...")
        splits = ['train', 'test']
        
        for split in splits:
            logging.info(f"\nProcessing {split} split...")
            # Set up input paths for this split
            annotation_dir = os.path.join(self.config.input_dir, f'Annotations_{split}')
            image_dir = os.path.join(self.config.input_dir, f'JPEGImages_{split}')
            
            # Verify directories exist
            if not os.path.exists(annotation_dir) or not os.path.exists(image_dir):
                logging.error(f"SSDD {split} directories not found: \n"
                            f"Annotation dir: {annotation_dir} ({os.path.exists(annotation_dir)})\n"
                            f"Image dir: {image_dir} ({os.path.exists(image_dir)})")
                continue
            
            logging.info(f"Found valid directories for {split} split")
            logging.info(f"Annotation directory: {annotation_dir}")
            logging.info(f"Image directory: {image_dir}")
            
            # Create output directories for this split
            split_output_dir = os.path.join(self.config.output_dir, split)
            split_image_dir = os.path.join(split_output_dir, 'images')
            split_label_dir = os.path.join(split_output_dir, 'labels')
            os.makedirs(split_image_dir, exist_ok=True)
            os.makedirs(split_label_dir, exist_ok=True)
            
            # Process each annotation file
            xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
            logging.info(f"Found {len(xml_files)} XML files to process")
            
            for xml_filename in xml_files:
                # logging.info(f"\nProcessing annotation file: {xml_filename}")
                try:
                    xml_path = os.path.join(annotation_dir, xml_filename)
                    annotation = self._parse_ssdd_xml(xml_path)
                    
                    # # Debug annotation content
                    # logging.info(f"Parsed annotation: filename={annotation['filename']}, "
                    #         f"number of objects={len(annotation['objects'])}")
                    
                    # Construct image path
                    image_filename = annotation['filename']
                    image_path = os.path.join(image_dir, image_filename)
                    
                    if not os.path.exists(image_path):
                        logging.error(f"Image not found: {image_path}")
                        continue
                    
                    # Process the image and its annotations
                    image = Image.open(image_path)
                    original_width, original_height = image.size
                    # logging.info(f"Loaded image: {image_path}, size={image.size}")
                    
                    # Debug objects coordinates
                    objects = annotation['objects']
                    # logging.info(f"Number of objects found: {len(objects)}")
                    # for i, obj in enumerate(objects):
                    #     logging.info(f"Object {i} coordinates: {obj['relative_coords']}")
                    
                    # Convert coordinates to relative if they're absolute
                    if objects and not all(0 <= coord <= 1 for obj in objects for coord in obj['relative_coords']):
                        # logging.info("Converting absolute coordinates to relative")
                        for obj in objects:
                            original_coords = obj['relative_coords']
                            obj['relative_coords'] = convert_absolute_to_relative(
                                original_coords, 
                                original_width, 
                                original_height
                            )
                            # logging.info(f"Converted coordinates:\n"
                            #         f"Original: {original_coords}\n"
                            #         f"Relative: {obj['relative_coords']}")
                    
                    if not objects:
                        logging.warning(f"No objects found in {xml_filename}")
                        continue
                    
                    # Process image and labels
                    relative_coords = [obj['relative_coords'] for obj in objects]
                    # logging.info(f"Processing {len(relative_coords)} labels")
                    # logging.info(f"Sample coordinates before processing: {relative_coords[0] if relative_coords else 'No coordinates'}")
                    
                    processed_img, processed_labels = process_image_and_labels(
                        image,
                        relative_coords,
                        self.config.target_size[0],
                        self.config.target_size[1]
                    )
                    
                    # logging.info(f"Processed labels count: {len(processed_labels)}")
                    # logging.info(f"Sample processed label: {processed_labels[0] if processed_labels else 'No processed labels'}")
                    
                    # Convert coordinates if absolute format requested
                    labels_to_save = processed_labels
                    if self.config.coord_format == CoordinateFormat.ABSOLUTE:
                        logging.info("Converting to absolute coordinates for saving")
                        labels_to_save = [
                            convert_relative_to_absolute(
                                label,
                                self.config.target_size[0],
                                self.config.target_size[1]
                            )
                            for label in processed_labels
                        ]
                    
                    # Save processed image
                    output_name = f"SSDD_{os.path.splitext(image_filename)[0]}"
                    output_image_path = os.path.join(split_image_dir, f"{output_name}.png")
                    processed_img.save(output_image_path)
                    # logging.info(f"Saved processed image to: {output_image_path}")
                    
                    # Save labels
                    output_label_path = os.path.join(split_label_dir, f"{output_name}.txt")
                    with open(output_label_path, 'w') as f:
                        for label in labels_to_save:
                            label_line = f"0 {' '.join(map(str, label))}\n"
                            f.write(label_line)
                            logging.debug(f"Wrote label: {label_line.strip()}")
                    
                    # logging.info(f"Saved {len(labels_to_save)} labels to: {output_label_path}")
                    
                    # Visualize if requested
                    if self.config.visualize:
                        vis_img = visualize_labels(
                            processed_img,
                            [[0] + list(coords) for coords in processed_labels]
                        )
                        vis_dir = os.path.join(split_output_dir, 'visualizations')
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_path = os.path.join(vis_dir, f"{output_name}_vis.png")
                        vis_img.save(vis_path)
                        # logging.info(f"Saved visualization to: {vis_path}")
                        
                except Exception as e:
                    logging.error(f"Error processing {xml_filename}:", exc_info=True)
                    continue

        logging.info("Completed SSDD dataset processing")

    def _process_single_image(
        self,
        image_info: Dict[str, Any],
        annotations: List[Any],
        dataset_name: str,
        convert_fn: Optional[Callable] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """Process a single image and its annotations"""
        current_output_dir = output_dir if output_dir is not None else self.config.output_dir
        
        # Get just the filename from the full path if it exists
        if 'file_name' in image_info:
            if os.path.isabs(image_info['file_name']):
                image_path = image_info['file_name']
                base_filename = os.path.basename(image_info['file_name'])
            else:
                # Handle different dataset structures
                if dataset_name == "SARShip":
                    # SAR-Ship-Dataset has images directly in input_dir
                    image_path = os.path.join(self.config.input_dir, image_info['file_name'])
                else:
                    # Other datasets have images in 'images' subdirectory
                    image_path = os.path.join(self.config.input_dir, 'images', image_info['file_name'])
                base_filename = image_info['file_name']
        else:
            logging.error("No file_name in image_info")
            return

        logging.info(f"Processing image: {image_path}")
        
        try:
            image = Image.open(image_path)
            original_width, original_height = image.size
        except Exception as e:
            logging.error(f"Failed to open image {image_path}: {str(e)}")
            return
            
        try:
            # Convert annotations to DOTA format if needed
            if dataset_name == "HRSID":
                # For HRSID, convert from absolute COCO to relative DOTA
                bbox_list = []
                for ann in annotations:
                    dota_abs = convert_coco_to_dota(ann['bbox'])
                    dota_rel = convert_absolute_to_relative(dota_abs, original_width, original_height)
                    bbox_list.append(dota_rel)
            elif convert_fn:
                bbox_list = [convert_fn(ann['bbox'] if isinstance(ann, dict) else ann[1:]) 
                            for ann in annotations]
            else:
                bbox_list = [ann['relative_coords'] if isinstance(ann, dict) else ann[1:]
                            for ann in annotations]
                
            # Process image and labels
            processed_img, processed_labels = process_image_and_labels(
                image, 
                bbox_list,
                self.config.target_size[0],
                self.config.target_size[1]
            )
            
            # Generate output paths using just the basename
            output_name = f"{dataset_name}_{os.path.splitext(base_filename)[0]}"
            
            # Setup output directories
            output_image_dir = os.path.join(current_output_dir, 'images')
            output_label_dir = os.path.join(current_output_dir, 'labels')
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)
            
            # Convert coordinates for saving if absolute format requested
            labels_to_save = processed_labels
            if self.config.coord_format == CoordinateFormat.ABSOLUTE:
                labels_to_save = [
                    convert_relative_to_absolute(label, self.config.target_size[0], self.config.target_size[1])
                    for label in processed_labels
                ]
            
            # Save labels
            with open(os.path.join(output_label_dir, f"{output_name}.txt"), 'w') as f:
                if dataset_name == "SRSDD":
                    for idx, label_coords in enumerate(labels_to_save):
                        class_id = annotations[idx].get('class_id', 0)  # Keep SRSDD class handling
                        f.write(f"{class_id} {' '.join(map(str, label_coords))}\n")
                else:
                    for label in labels_to_save:
                        f.write(f"0 {' '.join(map(str, label))}\n")
            
            # Save processed image
            processed_img.save(os.path.join(output_image_dir, f"{output_name}.png"))
            
            # Visualize if requested
            if self.config.visualize:
                # Prepare visualization labels with correct class IDs
                if dataset_name == "SRSDD":
                    vis_labels = [[ann.get('class_id', 0)] + list(coords) 
                                for ann, coords in zip(annotations, processed_labels)]
                else:
                    vis_labels = [[0] + list(label) for label in processed_labels]
                
                vis_img = visualize_labels(
                    processed_img,
                    vis_labels,
                )
                vis_dir = os.path.join(current_output_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                vis_img.save(os.path.join(vis_dir, f"{output_name}_vis.png"))
                
        except Exception as e:
            logging.error(f"Error processing annotations for {image_path}: {str(e)}")
            logging.error(f"Details: {str(e)}", exc_info=True)

    @staticmethod
    def _parse_ssdd_xml(xml_path: str) -> Dict[str, Any]:
        """
        Parse SSDD XML annotation file with rotated bounding boxes.
        
        Args:
            xml_path: Path to XML annotation file.
            
        Returns:
            Dictionary containing:
                - filename: image filename
                - width: image width
                - height: image height
                - objects: list of dictionaries with relative_coords
                
        Example XML structure:
            <annotation>
                <filename>000051.jpg</filename>
                <size>
                    <width>410</width>
                    <height>306</height>
                </size>
                <object>
                    <rotated_bndbox>
                        <x1>89</x1>
                        <y1>38</y1>
                        ...
                    </rotated_bndbox>
                </object>
            </annotation>
        """
        # logging.info(f"Parsing XML file: {xml_path}")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract image information
            filename = root.find('filename').text
            if filename is None:
                raise ValueError(f"Missing filename in {xml_path}")
            
            # Extract image size
            size_elem = root.find('size')
            if size_elem is None:
                raise ValueError(f"Missing size information in {xml_path}")
                
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            # logging.info(f"Image size: {width}x{height}")
            
            # Extract object information
            objects = []
            for i, obj in enumerate(root.findall('object')):
                try:
                    # Find rotated bounding box
                    bbox = obj.find('rotated_bndbox')
                    if bbox is None:
                        logging.warning(f"No rotated_bndbox found in object {i}")
                        continue
                    
                    # Extract coordinates
                    coords = []
                    for point in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
                        coord_elem = bbox.find(point)
                        if coord_elem is None or coord_elem.text is None:
                            raise ValueError(f"Missing {point} coordinate")
                        coords.append(float(coord_elem.text))
                    
                    # Convert to relative coordinates
                    rel_coords = convert_absolute_to_relative(coords, width, height)
                    # logging.info(f"Object {i} - Absolute coords: {coords}")
                    # logging.info(f"Object {i} - Relative coords: {rel_coords}")
                    
                    objects.append({'relative_coords': rel_coords})
                    
                except (AttributeError, ValueError) as e:
                    logging.warning(f"Invalid coordinates in object {i}: {str(e)}")
                    continue
            
            # logging.info(f"Successfully parsed {len(objects)} objects")
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            }
            
        except ET.ParseError as e:
            logging.error(f"Failed to parse XML: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error parsing XML: {str(e)}")
            raise