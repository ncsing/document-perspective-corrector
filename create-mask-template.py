import argparse
import cv2
import numpy as np

# Move the following variables to the global scope
ix, iy, drawing, img, rectangles = -1, -1, False, None, []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, rectangles

    global ix, iy, drawing, img, rectangles

    if img is None:
        print("Error: Image not loaded")
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 255, 255), -1)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (255, 255, 255), -1)
        rectangles.append((ix, iy, x-ix, y-iy))
        cv2.imshow('image', img)

# Load and resize the image
def load_image(path):
    global img
    original_img = cv2.imread(path)
    if original_img is None:
        print(f"Error: Unable to load image from {path}")
        return False
    img = cv2.resize(original_img, (600, 400))
    return True

def process_image():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    argument_parser.add_argument("-o", "--output", required=True, help="Path to the rectangle locations file")
    arguments = vars(argument_parser.parse_args())

    # Read and resize the image
    if not load_image(arguments["image"]):
        exit(1)

    # Create a window and set mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    global drawing, ix, iy, rectangles

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

    # Print the rectangle locations
    print("Rectangle locations (x, y, width, height):")
    for rect in rectangles:
        print(rect)

    # # Save the masked image
    # cv2.imwrite('masked_image.jpg', img)

    # Save rectangle locations to a file
    with open(arguments["output"], 'w') as f:
        for rect in rectangles:
            f.write(f"{rect[0]},{rect[1]},{rect[2]},{rect[3]}\n")

if __name__ == "__main__":
    process_image()
