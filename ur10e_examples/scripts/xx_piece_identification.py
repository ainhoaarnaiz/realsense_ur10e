from keras.models import load_model
import cv2
import numpy as np

model_path = '/dev_ws/src/ur10e_examples/model/keras_model.h5'
label_path = '/dev_ws/src/ur10e_examples/model/labels.txt'
image_path = '/dev_ws/src/ur10e_examples/model/label_02.jpeg'

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(model_path, compile=False)

# Load the labels
class_names = open(label_path, "r").readlines()

# Read the uploaded image
image = cv2.imread(image_path)

# Resize the raw image into (224-height,224-width) pixels
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# Show the image in a window
# cv2.imshow("Uploaded Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Make the image a numpy array and reshape it to the model's input shape
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

# Normalize the image array
image = (image / 127.5) - 1

# Predict the model
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
