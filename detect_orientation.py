# import the necessary packages
from pytesseract import Output
import pytesseract
import imutils
import cv2

# Hard-code the path to your image here
image_path = "emcheck.png"

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image  # No resizing needed

    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

# display the orientation information
print("[INFO] detected orientation: {}".format(results["orientation"]))
print("[INFO] rotate by {} degrees to correct".format(results["rotate"]))
print("[INFO] detected script: {}".format(results["script"]))

# rotate the image to correct the orientation
rotated = imutils.rotate_bound(image, angle=results["rotate"])

resized_original = resize_image(image, width=800)  # Adjust width as needed
resized_rotated = resize_image(rotated, width=800)  # Adjust width as needed

# Show the original image and output image after orientation correction
cv2.imshow("Original", resized_original)
cv2.imshow("Output", resized_rotated)

# Save the corrected image
cv2.imwrite("emcheck_corrected.png", rotated)

cv2.waitKey(0)
