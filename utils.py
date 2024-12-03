from typing import List, Tuple, Union, Optional
import warnings
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants for visualization
DEFAULT_BOX_COLOR = (255, 0, 0)  # Red
DEFAULT_TEXT_COLOR = (255, 255, 255)  # White
DEFAULT_LINE_THICKNESS = 2
DEFAULT_FONT_SIZE = 12
MIN_LINE_THICKNESS = 1
MAX_LINE_THICKNESS = 10
RANDOM_SEED = 42

def validate_rgb_color(color: Tuple[int, int, int]) -> bool:
    """
    Validates RGB color tuple.
    
    Args:
        color: Tuple of (R,G,B) values.
    
    Returns:
        bool: True if valid RGB color, False otherwise.
    """
    if not isinstance(color, tuple) or len(color) != 3:
        return False
    return all(isinstance(v, int) and 0 <= v <= 255 for v in color)

def validate_coordinates(coords: List[float]) -> bool:
    """
    Validates coordinate list format.
    
    Args:
        coords: List of relative coordinates.
    
    Returns:
        bool: True if valid coordinates, False otherwise.
    """
    if not isinstance(coords, list):
        return False
    if len(coords) % 2 != 0:  # Must have pairs of coordinates
        return False
    return all(isinstance(v, (int, float)) for v in coords)

def convert_relative_to_absolute(
    coords: List[float], 
    img_width: int, 
    img_height: int
) -> List[int]:
    """
    Converts relative coordinates (0-1) to absolute pixel coordinates.
    
    Args:
        coords: List of relative coordinates (x1, y1, x2, y2, ...) between 0 and 1.
        img_width: Image width in pixels for scaling.
        img_height: Image height in pixels for scaling.
    
    Returns:
        List of absolute pixel coordinates, clamped to image boundaries if necessary.
    
    Raises:
        TypeError: If image dimensions are not integers.
        ValueError: If image dimensions are not positive.
    
    Examples:
        >>> coords = [0.1, 0.2, 0.3, 0.4]  # [x1, y1, x2, y2]
        >>> img_width, img_height = 100, 200
        >>> convert_relative_to_absolute(coords, img_width, img_height)
        [10, 40, 30, 80]  # Converted to pixel coordinates
        
        >>> # Handling out-of-bounds coordinates
        >>> coords = [-0.1, 1.2, 0.3, 0.4]
        >>> convert_relative_to_absolute(coords, img_width, img_height)
        [0, 200, 30, 80]  # Clamped to image boundaries
    """
    # Validate inputs
    if not validate_coordinates(coords):
        raise ValueError("Invalid coordinate format")
    
    if not all(0 <= coord <= 1 for coord in coords):
        warnings.warn("Some relative coordinates are out of bounds (not between 0 and 1).")
    
    if not (isinstance(img_width, int) and isinstance(img_height, int)):
        raise TypeError("Image dimensions must be integers")
    
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # Adjust out-of-bound relative coordinates and issue warnings
    for i in range(len(coords)):
        if coords[i] < 0:
            warnings.warn(f"Coordinate {coords[i]} is less than 0. Clamping to 0.")
            coords[i] = 0
        elif coords[i] > 1:
            warnings.warn(f"Coordinate {coords[i]} is greater than 1. Clamping to 1.")
            coords[i] = 1

    # Convert relative coordinates to absolute coordinates
    absolute_coords = [
        round(coords[i] * (img_width if i % 2 == 0 else img_height))
        for i in range(len(coords))
    ]

    # Ensure absolute coordinates fit within image boundaries (clamp to min/max)
    for i in range(len(absolute_coords)):
        if i % 2 == 0:  # x-coordinate
            if absolute_coords[i] < 0:
                warnings.warn(f"x-coordinate {absolute_coords[i]} is less than 0. Clamping to 0.")
                absolute_coords[i] = 0
            elif absolute_coords[i] > img_width:
                warnings.warn(f"x-coordinate {absolute_coords[i]} exceeds image width. Clamping to {img_width}.")
                absolute_coords[i] = img_width
        else:  # y-coordinate
            if absolute_coords[i] < 0:
                warnings.warn(f"y-coordinate {absolute_coords[i]} is less than 0. Clamping to 0.")
                absolute_coords[i] = 0
            elif absolute_coords[i] > img_height:
                warnings.warn(f"y-coordinate {absolute_coords[i]} exceeds image height. Clamping to {img_height}.")
                absolute_coords[i] = img_height

    return absolute_coords


def convert_absolute_to_relative(
    coords: List[int], 
    img_width: int, 
    img_height: int
) -> List[float]:
    """
    Converts absolute pixel coordinates to relative coordinates (0-1).
    
    Args:
        coords: List of absolute pixel coordinates (x1, y1, x2, y2, ...).
        img_width: Image width in pixels.
        img_height: Image height in pixels.
    
    Returns:
        List of relative coordinates between 0 and 1, rounded to 6 decimal places.
    
    Raises:
        TypeError: If image dimensions are not integers.
        ValueError: If image dimensions are not positive.
    
    Examples:
        >>> coords = [10, 40, 30, 80]  # [x1, y1, x2, y2] in pixels
        >>> img_width, img_height = 100, 200
        >>> convert_absolute_to_relative(coords, img_width, img_height)
        [0.1, 0.2, 0.3, 0.4]  # Converted to relative coordinates
        
        >>> # Handling out-of-bounds coordinates
        >>> coords = [-10, 250, 30, 80]
        >>> convert_absolute_to_relative(coords, img_width, img_height)
        [0.0, 1.0, 0.3, 0.4]  # Clamped to valid range
    """
    # Validate input
    if not (isinstance(img_width, int) and isinstance(img_height, int)):
        raise TypeError("Image dimensions must be integers")
    
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # Adjust out-of-bound coordinates and issue warnings
    for i in range(len(coords)):
        if i % 2 == 0:  # x-coordinate (even index)
            if coords[i] < 0:
                warnings.warn(f"x-coordinate {coords[i]} is out of bounds (less than 0). Clamping to 0.")
                coords[i] = 0
            elif coords[i] > img_width:
                warnings.warn(f"x-coordinate {coords[i]} is out of bounds (greater than image width). Clamping to {img_width}.")
                coords[i] = img_width
        else:  # y-coordinate (odd index)
            if coords[i] < 0:
                warnings.warn(f"y-coordinate {coords[i]} is out of bounds (less than 0). Clamping to 0.")
                coords[i] = 0
            elif coords[i] > img_height:
                warnings.warn(f"y-coordinate {coords[i]} is out of bounds (greater than image height). Clamping to {img_height}.")
                coords[i] = img_height
    
    # Convert absolute coordinates to relative coordinates
    return [
        round(coords[i] / (img_width if i % 2 == 0 else img_height), 6)
        for i in range(len(coords))
    ]


def convert_yolo_to_dota(yolo_bbox: List[float]) -> List[float]:
    """
    Converts YOLO format bounding box to DOTA format coordinates.
    
    Args:
        yolo_bbox: YOLO format bounding box [x_center, y_center, width, height]
                  where all values are relative (0-1).
    
    Returns:
        List of 8 coordinates in DOTA format [x1, y1, x2, y2, x3, y3, x4, y4]
        representing the four corners in clockwise order starting from top-left.
    
    Examples:
        >>> yolo_bbox = [0.5, 0.5, 0.2, 0.3]  # center_x, center_y, width, height
        >>> convert_yolo_to_dota(yolo_bbox)
        [0.4, 0.35, 0.6, 0.35, 0.6, 0.65, 0.4, 0.65]  # Four corners clockwise
    """
    center_x, center_y, width, height = yolo_bbox
    
    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2
    
    # Calculate corners (clockwise from top-left)
    x1, y1 = center_x - half_width, center_y - half_height  # top-left
    x2, y2 = center_x + half_width, center_y - half_height  # top-right
    x3, y3 = center_x + half_width, center_y + half_height  # bottom-right
    x4, y4 = center_x - half_width, center_y + half_height  # bottom-left
    
    return [x1, y1, x2, y2, x3, y3, x4, y4]

def convert_coco_to_dota(coco_bbox: List[float]) -> List[float]:
    """
    Converts COCO format bounding box to DOTA format coordinates.
    
    Args:
        coco_bbox: COCO format bounding box [x_min, y_min, width, height]
                  where all values are in absolute coordinates.
    
    Returns:
        List of 8 coordinates in DOTA format [x1, y1, x2, y2, x3, y3, x4, y4]
        representing the four corners in clockwise order.
    
    Examples:
        >>> coco_bbox = [100, 100, 50, 30]  # x_min, y_min, width, height
        >>> convert_coco_to_dota(coco_bbox)
        [100, 100, 150, 100, 150, 130, 100, 130]  # Four corners
    """
    x_min, y_min, width, height = coco_bbox
    x1, y1 = x_min, y_min
    x2, y2 = x_min + width, y_min
    x3, y3 = x_min + width, y_min + height
    x4, y4 = x_min, y_min + height
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def check_image_size(
    img_width: int, 
    img_height: int, 
    target_width: int, 
    target_height: int
) -> str:
    """
    Validates image dimensions against target size and determines if padding is needed.
    
    Args:
        img_width: Current image width in pixels.
        img_height: Current image height in pixels.
        target_width: Desired target width in pixels.
        target_height: Desired target height in pixels.
    
    Returns:
        'valid' if image matches target size, 'pad' if padding is needed.
    
    Raises:
        ValueError: If image size exceeds target size.
    
    Examples:
        >>> check_image_size(800, 600, 1024, 1024)
        'pad'
        >>> check_image_size(1024, 1024, 1024, 1024)
        'valid'
        >>> check_image_size(1200, 1024, 1024, 1024)
        ValueError: Image size (1200x1024) exceeds target size (1024x1024).
    """
    if img_width > target_width or img_height > target_height:
        raise ValueError(f"Image size ({img_width}x{img_height}) exceeds target size ({target_width}x{target_height}).")
    elif img_width < target_width or img_height < target_height:
        return 'pad'
    return 'valid'


def random_pad_image_and_rel_labels(
    image: Image.Image,
    rel_labels: List[List[float]],
    target_width: int,
    target_height: int,
    random_seed: int = RANDOM_SEED
) -> Tuple[Image.Image, List[List[float]]]:
    
    """
    Randomly pads an image to target size and adjusts relative label coordinates.
    Uses fixed random seed for reproducibility.
    
    Args:
        image: PIL Image object to be padded.
        rel_labels: List of relative coordinate labels in DOTA format.
                   Each label is [x1, y1, x2, y2, x3, y3, x4, y4].
        target_width: Desired width after padding.
        target_height: Desired height after padding.
    
    Returns:
        Tuple of (padded_image, updated_labels) where updated_labels maintains
        the relative coordinate format but adjusted for padding.
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    img_width, img_height = image.size

    # Calculate padding
    pad_width = target_width - img_width
    pad_height = target_height - img_height
    left = random.randint(0, pad_width)
    right = pad_width - left
    top = random.randint(0, pad_height)
    bottom = pad_height - top

    # Pad the image
    padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    # Update label
    updated_rel_labels = []
    # print(rel_labels)
    
    for rel_label in rel_labels:
        abs_label = convert_relative_to_absolute(rel_label, img_width, img_height)

        # Adjust absolute coordinates for padding
        abs_label = [
            abs_label[0] + left, abs_label[1] + top,
            abs_label[2] + left, abs_label[3] + top,
            abs_label[4] + left, abs_label[5] + top,
            abs_label[6] + left, abs_label[7] + top,
        ]

        # Convert back to relative coordinates
        padded_rel_label = convert_absolute_to_relative(abs_label, target_width, target_height)
        updated_rel_labels.append(padded_rel_label)

    return padded_image, updated_rel_labels

def process_image_and_labels(
    image: Image.Image,
    labels: List[List[float]],
    target_width: int,
    target_height: int
) -> Tuple[Image.Image, List[List[float]]]:
    """
    Processes an image and its labels to match target size requirements.
    
    Args:
        image: PIL Image object to process.
        labels: List of relative coordinate labels in DOTA format.
        target_width: Desired width for processed image.
        target_height: Desired height for processed image.
    
    Returns:
        Tuple of (processed_image, processed_labels) where processed_image
        matches target dimensions and labels are adjusted accordingly.
    
    Examples:
        >>> img = Image.new('RGB', (800, 600))
        >>> labels = [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]]
        >>> new_img, new_labels = process_image_and_labels(
        ...     img, labels, 1024, 1024)
        >>> new_img.size
        (1024, 1024)
    """
    img_width, img_height = image.size
    status = check_image_size(img_width, img_height, target_width, target_height)

    if status == 'pad':
        return random_pad_image_and_rel_labels(image, labels, target_width, target_height)
    return image, labels

def visualize_labels(
    image: Union[str, Image.Image],
    labels: List[List[float]],
    output_path: Optional[str] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> Image.Image:
    """
    Visualizes bounding box labels on an image. Handles both RGB and grayscale images.
    
    Args:
        image: PIL Image object or path to image file.
        labels: List of labels in format [class_idx, x1, y1, x2, y2, x3, y3, x4, y4]
               where coordinates are relative (0-1).
        output_path: Optional path to save the visualized image.
        color: RGB color tuple for drawing boxes, default is red (255, 0, 0).
        thickness: Line thickness for boxes in pixels, default is 2.
    
    Returns:
        PIL Image with visualized bounding boxes.
    
    Raises:
        ValueError: If color is not a valid RGB tuple or thickness is out of range.
        IOError: If image file cannot be opened or saved.
    """
    # Validate inputs
    if not validate_rgb_color(color):
        raise ValueError("Invalid RGB color tuple")
    
    if not isinstance(thickness, int) or thickness < MIN_LINE_THICKNESS or thickness > MAX_LINE_THICKNESS:
        raise ValueError(f"Line thickness must be between {MIN_LINE_THICKNESS} and {MAX_LINE_THICKNESS}")
    
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
    except Exception as e:
        logging.error(f"Failed to open image: {str(e)}")
        raise IOError(f"Could not open image: {str(e)}")
    
    # Convert grayscale to RGB for visualization
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB')
        logging.warning(f"Converting image from {image.mode} to RGB for visualization")
    
    # Create a copy of the image to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    img_width, img_height = image.size
    
    # Process each label
    for label in labels:
        # Validate label format
        if not label or len(label) < 9:
            logging.warning(f"Skipping invalid label: {label}")
            continue
            
        try:
            # Extract class index and coordinates
            class_idx = int(label[0])
            coords = label[1:]
            
            # Validate coordinates
            if not validate_coordinates(coords):
                logging.warning(f"Skipping label with invalid coordinates: {coords}")
                continue
            
            # Convert relative coordinates to absolute
            abs_coords = convert_relative_to_absolute(coords, img_width, img_height)
            
            # Draw the bounding box
            points = [
                (abs_coords[0], abs_coords[1]),
                (abs_coords[2], abs_coords[3]),
                (abs_coords[4], abs_coords[5]),
                (abs_coords[6], abs_coords[7]),
                (abs_coords[0], abs_coords[1])  # Close the polygon
            ]
            
            # Draw lines connecting the points
            for i in range(4):
                draw.line([points[i], points[i+1]], fill=color, width=thickness)
                
            # Add class label with background for better visibility
            label_pos = (abs_coords[0], abs_coords[1] - 20)
            label_text = f"Class {class_idx}"
            
            # Draw text background
            text_bbox = draw.textbbox(label_pos, label_text)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                         fill=(0, 0, 0))  # Black background
            
            # Draw text
            draw.text(label_pos, label_text, fill=DEFAULT_TEXT_COLOR)
            
        except Exception as e:
            logging.error(f"Error processing label: {str(e)}")
            continue
    
    # Save if output path provided
    if output_path:
        try:
            img_draw.save(output_path)
        except Exception as e:
            logging.error(f"Failed to save image: {str(e)}")
            raise IOError(f"Could not save image: {str(e)}")
    
    return img_draw