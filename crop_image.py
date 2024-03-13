from pytesseract import Output
import pytesseract
import cv2
import numpy as np
import imutils

# Function to automatically determine text orientation and rotate the image to correct orientation
def rotate_image(image_data):
    # Use Tesseract to get the orientation of the text
    orientation_details = pytesseract.image_to_osd(image_data, output_type=Output.DICT)
    print(f"[INFO] Detected orientation: {orientation_details['orientation']}")
    print(f"[INFO] Rotate by {orientation_details['rotate']} degrees to correct")
    print(f"[INFO] Detected script: {orientation_details['script']}")

    # Rotate the image to correct the orientation
    corrected_image = imutils.rotate_bound(image_data, angle=orientation_details['rotate'])
    return corrected_image

# Function to locate the check within the image, correct its perspective, and crop it
def locate_and_crop_check(input_image_path):
    original_image = cv2.imread(input_image_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Improve edge detection by blurring and using Canny edge detector
    blurred_gray = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edges_detected = cv2.Canny(blurred_gray, 50, 150)

    # Find contours from the edges, we assume the largest contour with 4 sides is the check
    found_contours, _ = cv2.findContours(edges_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(original_image, found_contours, -1, (0, 255, 0), 2)
    # cv2.imshow("Contours", original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Loop over the contours to identify potential checks
    for contour in found_contours:
        arc_length = cv2.arcLength(contour, True)
        contour_approximation = cv2.approxPolyDP(contour, 0.02 * arc_length, True)

        if len(contour_approximation) == 4 and cv2.contourArea(contour) > 1000:
            check_contour = cv2.minAreaRect(contour_approximation)
            check_box_points = cv2.boxPoints(check_contour)
            check_box_points = np.intp(check_box_points)

            # Perspective transform to get a top-down view of the check
            check_width = int(check_contour[1][0])
            check_height = int(check_contour[1][1])
            source_points = check_box_points.astype("float32")
            destination_points = np.array([[0, check_height-1], [0, 0], [check_width-1, 0], [check_width-1, check_height-1]], dtype="float32")

            transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            corrected_image = cv2.warpPerspective(original_image, transformation_matrix, (check_width, check_height))
            
            # Auto-rotate the cropped check image for correct text orientation
            try:
                final_image = rotate_image(corrected_image)
                cv2.imshow("Corrected Check Orientation", final_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return final_image
            except pytesseract.TesseractError:
                print("Failed to auto-detect text orientation.")

    # If no check contour is found
    print("[INFO] No suitable check-like contour was found in the image.")
    return None

# Main function to execute the check orientation and cropping
def main():
    # Specify the path to your image
    check_image_path = "emcheck.png"
    oriented_and_cropped_check = locate_and_crop_check(check_image_path)

    # Save the oriented and cropped check image if found
    if oriented_and_cropped_check is not None:
        cv2.imwrite("emcheck_cropped.png", oriented_and_cropped_check)
        print("[INFO] Check image has been oriented and cropped successfully.")

if __name__ == "__main__":
    main()
