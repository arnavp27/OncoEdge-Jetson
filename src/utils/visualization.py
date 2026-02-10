"""
Visualization Utilities

Functions for drawing detection results on images.
"""
import cv2
import numpy as np


# Color mapping for different classes
CLASS_COLORS = {
    'OSCC': (255, 0, 0),      # Red for cancer
    'OPMD': (255, 255, 0),    # Yellow for pre-malignant
    'Normal': (0, 255, 0)     # Green for normal
}


def draw_detections(image, detections):
    """
    Draw bounding boxes, masks, and labels on image.

    Args:
        image: Original image (numpy array HxWx3, RGB)
        detections: List of detection dictionaries containing:
            - 'bbox': [x1, y1, x2, y2]
            - 'mask': Segmentation mask (optional)
            - 'class': Class name
            - 'fusion_score': Confidence score

    Returns:
        numpy array: Annotated image
    """
    # Create a copy to avoid modifying original
    vis_image = image.copy()

    # Convert RGB to BGR for OpenCV
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    for idx, detection in enumerate(detections):
        bbox = detection['bbox']
        class_name = detection['class']
        score = detection['fusion_score']

        # Get color for this class
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw mask overlay if available
        if 'mask' in detection and detection['mask'] is not None:
            mask = detection['mask']
            # Ensure mask is 2D
            if len(mask.shape) > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask.squeeze()

            # Resize mask to image size if needed
            if mask.shape != (vis_image.shape[0], vis_image.shape[1]):
                mask = cv2.resize(mask.astype(np.float32),
                                (vis_image.shape[1], vis_image.shape[0]))

            # Create colored mask overlay
            mask_binary = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(vis_image)
            colored_mask[:, :] = color

            # Blend mask with image
            vis_image = np.where(
                mask_binary[:, :, None] > 0,
                cv2.addWeighted(vis_image, 0.6, colored_mask, 0.4, 0),
                vis_image
            )

        # Draw label with background
        label = f"{class_name}: {score:.2%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            vis_image,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        # Add lesion number
        cv2.putText(
            vis_image,
            f"#{idx + 1}",
            (x1 + 5, y2 - 10),
            font,
            0.5,
            color,
            2
        )

    # Convert back to RGB for display
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    return vis_image


def create_risk_visualization(image, risk_level):
    """
    Add risk level indicator to image.

    Args:
        image: Image to annotate
        risk_level: Risk level string ("HIGH", "MEDIUM", "LOW")

    Returns:
        numpy array: Image with risk indicator
    """
    vis_image = image.copy()

    # Color based on risk
    risk_colors = {
        'HIGH': (255, 0, 0),
        'MEDIUM': (255, 255, 0),
        'LOW': (0, 255, 0)
    }

    color = risk_colors.get(risk_level, (255, 255, 255))

    # Convert to BGR for OpenCV
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    # Add risk banner at top
    banner_height = 50
    banner = np.zeros((banner_height, vis_image.shape[1], 3), dtype=np.uint8)
    banner[:] = color

    # Add text to banner
    text = f"RISK LEVEL: {risk_level}"
    font = cv2.FONT_HERSHEY_BOLD
    font_scale = 1.0
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (vis_image.shape[1] - text_width) // 2
    text_y = (banner_height + text_height) // 2

    cv2.putText(banner, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness)

    # Combine banner with image
    vis_image = np.vstack([banner, vis_image])

    # Convert back to RGB
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    return vis_image
