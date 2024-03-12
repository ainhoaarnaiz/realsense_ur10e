import cv2
import easyocr
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path
filename = os.path.join(current_directory, "label_02.jpeg")

# Read the image
image = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Adjust language and GPU usage as needed

# Perform text detection
result = reader.readtext(gray)

# Extract detected text
detected_text = [text[1] for text in result]

# Print the detected text
print("Detected Handwritten Text:")
for text in detected_text:
    print(text)
