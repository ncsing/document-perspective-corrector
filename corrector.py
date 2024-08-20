import cv2
import argparse
import imutils
import numpy as np

def calculate_corners(contour_points):
    corners = np.zeros((4, 2), dtype="float32")
    sum_points = contour_points.sum(axis=1)
    corners[0] = contour_points[np.argmin(sum_points)]
    corners[2] = contour_points[np.argmax(sum_points)]
    diff_points = np.diff(contour_points, axis=1)
    corners[1] = contour_points[np.argmin(diff_points)]
    corners[3] = contour_points[np.argmax(diff_points)]
    return corners

def transform_perspective(input_image, contour_points):
    corners = calculate_corners(contour_points)
    (top_left, top_right, bottom_right, bottom_left) = corners

    width_top = np.linalg.norm(bottom_right - bottom_left)
    width_bottom = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(top_right - bottom_right)
    height_right = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_left), int(height_right))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, destination)
    output_image = cv2.warpPerspective(input_image, matrix, (max_width, max_height))

    return output_image

def improve_edge_detection(image):
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to close gaps
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Use Canny edge detection with lower thresholds
    edges = cv2.Canny(morph, 30, 100)

    return edges

def process_image():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    argument_parser.add_argument("-o", "--output", required=True, help="Path to the output image file")
    argument_parser.add_argument("-m", "--mask", required=False, help="Path to the mask image file")
    arguments = vars(argument_parser.parse_args())

    original_image = cv2.imread(arguments["image"])
    image_ratio = original_image.shape[0] / 500.0
    resized_image = imutils.resize(original_image, height=500)

    edge_detected = improve_edge_detection(resized_image)

    # STEP 1: Edge Detection
    print("STEP 1: Edge Detection")
    cv2.imshow("STEP 1: Edges", edge_detected)

    contours = cv2.findContours(edge_detected.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    filtered_contours = []
    image_area = resized_image.shape[0] * resized_image.shape[1]
    min_area = image_area * 0.2
    max_area = image_area * 1
    for contour in sorted_contours:
        contour_area = cv2.contourArea(contour)
        if min_area <= contour_area <= max_area:
            filtered_contours.append(contour)
    print("Sorted Contours: ", len(sorted_contours))
    print("Filtered Contours: ", len(filtered_contours))

    detected_screen = None
    for contour in filtered_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Show the minimum area rectangle
        print("Minimum Area Rectangle Points: ", len(box))
        min_area_rect_image = np.zeros(resized_image.shape, dtype=np.uint8)
        cv2.drawContours(min_area_rect_image, [box], 0, (0, 255, 0), 2)
        cv2.imshow("STEP 2: Minimum Area Rectangle", min_area_rect_image)
        cv2.waitKey(0)

        detected_screen = box
        break
    
    if detected_screen is None:
        print("No valid screen contour found.")
        cv2.destroyAllWindows()
        return

    # STEP 2: Finding Boundary
    print("STEP 2: Finding Boundary")
    cv2.drawContours(resized_image, [detected_screen], -1, (0, 255, 0), 2)
    cv2.imshow("Boundary", resized_image)

    transformed_image = transform_perspective(original_image, detected_screen.reshape(4, 2) * image_ratio)
    grayscale_transformed = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

    # STEP 3: Apply Perspective Transform
    print("STEP 3: Apply Perspective Transform")
    cv2.imshow("Scanned Image", transformed_image)

    cv2.waitKey(0)

    
    is_mask = True if arguments["mask"] else False
    if is_mask is False:
        cv2.imwrite(arguments["output"], transformed_image)
        cv2.destroyAllWindows()
        return

    # STEP 4: Masking
    # Scale the transformed image to 600x400
    masked_image = cv2.resize(transformed_image, (600, 400))
    # Define regions to mask (x, y, width, height)
    regions_to_mask = []
    with open(arguments["mask"], 'r') as f:
        for line in f:
            x, y, w, h = map(int, line.strip().split(','))
            regions_to_mask.append((x, y, w, h))
    # Create a white rectangle for each region
    for (x, y, w, h) in regions_to_mask:
        cv2.rectangle(masked_image, (x, y), (x+w, y+h), (255, 255, 255), -1)

    cv2.imwrite(arguments["output"], masked_image)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image()
