import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Load the image
image_path = "emcheck_cropped.png"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Tesseract to find the word "date"
custom_config = r'--oem 3 --psm 6'
d = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)
n_boxes = len(d['text'])
print("Detected text:", d['text'])

# # Initialize variables to store the bounding box of the word "date"
# date_x, date_y, date_w, date_h = 0, 0, 0, 0
# for i in range(n_boxes):
#     if d['text'][i].lower() == 'date':
#         (date_x, date_y, date_w, date_h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         break
# Initialize variables to store the bounding box of the word "date"
date_x, date_y, date_w, date_h = None, None, None, None
for i, text in enumerate(d['text']):
    if "date" in text.lower():
        # Extract the bounding box coordinates
        date_x, date_y, date_w, date_h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        if text.lower().endswith(','):
            date_w -= 10  # Adjust width to exclude comma
        break

# Draw the bounding box around "date" if it was found
if date_x is not None and date_y is not None and date_w is not None and date_h is not None:
    cv2.rectangle(img, (date_x, date_y), (date_x + date_w, date_y + date_h), (0, 255, 0), 2)
    print("Bounding box for 'date':", date_x, date_y, date_w, date_h)
else:
    print("Could not find 'date' in the image.")

# Draw the bounding box around "date"
cv2.rectangle(img, (date_x, date_y), (date_x + date_w, date_y + date_h), (0, 255, 0), 2)
print("Bounding box for 'date':", date_x, date_y, date_w, date_h)

# Perform edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use Hough Line Transform to find lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
print("Number of lines detected:", len(lines))

# Variables to store the best line's parameters
best_line = None
min_y_distance = float('inf')

# Iterate over the lines
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check if the line is within the x-range of the "date" bounding box
            if x1 >= date_x and x2 <= date_x + date_w:
                # Calculate the vertical distance to the bottom of the "date" bounding box
                y_distance = abs(y1 - (date_y + date_h))
                # Check if this line is closer to the bottom of the "date" box than previous lines
                if y_distance < min_y_distance:
                    min_y_distance = y_distance
                    best_line = (x1, y1, x2, y2)

# Draw the best line on the image
if best_line is not None:
    cv2.line(img, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (255, 0, 0), 2)
print("Best line parameters:", best_line)

# Draw the best line on the image if one was found
if best_line:
    cv2.line(img, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (255, 0, 0), 2)
    print("Best line parameters:", best_line)
else:
    print("No suitable line found for 'date'")

# Save or display the image
cv2.imwrite('emcheck_date_line.png', img)
cv2.imshow('Date Line', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
