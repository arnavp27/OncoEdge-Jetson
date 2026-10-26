"""
Visualization Utilities

Functions for drawing detection results on images.
"""
import cv2
import numpy as np


# OpenCV draws in BGR; returned images are converted back to RGB.
CLASS_COLORS = {
    'OCA': (0, 0, 255),
    'OSCC': (0, 0, 255),
    'OPMD': (0, 255, 255),
    'Benign': (255, 200, 0),
    'Normal': (0, 200, 0),
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
            - 'fusion_score': Uncalibrated semantic match score

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

    # Draw labels after all masks so a later region cannot tint or hide them.
    label_rects = []
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class']
        score = detection['fusion_score']
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        needs_review = (detection.get('model_disagreement') or detection.get('classification_uncertain')
                        or detection.get('detection_uncertain'))
        label = f"#{idx+1} {'? ' if needs_review else ''}{class_name} | match {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        if text_width + 10 > vis_image.shape[1]:
            font_scale *= (vis_image.shape[1] - 10) / (text_width + 10)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
        label_x = min(max(0, x1), max(0, vis_image.shape[1] - text_width - 10))
        label_y = min(max(text_height + 10, y1), vis_image.shape[0] - baseline - 1)

        # Overlapping boxes often share their top edge; move subsequent labels
        # down a row instead of covering the earlier model output.
        while True:
            rect = (label_x, label_y - text_height - 10,
                    label_x + text_width + 10, label_y + baseline)
            overlaps = any(
                rect[0] < other[2] and rect[2] > other[0]
                and rect[1] < other[3] and rect[3] > other[1]
                for other in label_rects
            )
            next_y = label_y + text_height + baseline + 12
            if not overlaps or next_y > vis_image.shape[0] - baseline - 1:
                break
            label_y = next_y
        label_rects.append(rect)

        # Draw background rectangle
        cv2.rectangle(
            vis_image,
            (label_x, label_y - text_height - 10),
            (label_x + text_width + 10, label_y + baseline),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            vis_image,
            label,
            (label_x + 5, label_y - 5),
            font,
            font_scale,
            (10, 10, 10),
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
