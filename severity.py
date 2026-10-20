import cv2
import numpy as np

def verify_and_analyze_leaf(image_bgr):
    """
    Analyzes crop leaf foliage using deterministic OpenCV HSV segmentation.
    Returns: (is_leaf, severity_pct, leaf_mask, lesion_mask, overlay_bgr)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. Broad foliage mask (greens, yellowish greens, dried margins)
    lower_plant = np.array([18, 25, 25])
    upper_plant = np.array([95, 255, 255])
    plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)
    
    # Noise cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    leaf_pixels = cv2.countNonZero(plant_mask)
    
    # If leaf presence is less than 4% of frame, reject as non-leaf
    if leaf_pixels < (total_pixels * 0.04):
        return False, 0.0, plant_mask, np.zeros_like(plant_mask), image_bgr

    # 2. Extract necrotic lesions (browns, dark rust spots, scorched tissue)
    lower_lesion1 = np.array([5, 50, 20])
    upper_lesion1 = np.array([24, 255, 180])
    lower_lesion2 = np.array([0, 40, 20])
    upper_lesion2 = np.array([12, 255, 140])
    
    mask_l1 = cv2.inRange(hsv, lower_lesion1, upper_lesion1)
    mask_l2 = cv2.inRange(hsv, lower_lesion2, upper_lesion2)
    raw_lesion = cv2.bitwise_or(mask_l1, mask_l2)
    
    # Constrain lesions strictly within the detected leaf boundaries
    lesion_mask = cv2.bitwise_and(raw_lesion, plant_mask)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    lesion_pixels = cv2.countNonZero(lesion_mask)
    severity_pct = round((lesion_pixels / float(leaf_pixels)) * 100.0, 2)
    
    # 3. Create high-visibility diagnostic overlay
    overlay_bgr = image_bgr.copy()
    
    # Red highlight over necrotic lesions
    red_layer = np.zeros_like(image_bgr)
    red_layer[:, :] = (0, 0, 230)
    lesion_indices = lesion_mask > 0
    overlay_bgr[lesion_indices] = cv2.addWeighted(image_bgr[lesion_indices], 0.35, red_layer[lesion_indices], 0.65, 0)
    
    # Yellow outer contour around leaf perimeter
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 235, 255), 2)
    
    return True, severity_pct, plant_mask, lesion_mask, overlay_bgr
