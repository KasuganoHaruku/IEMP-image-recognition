from skimage import feature, exposure
import cv2
import joblib
from PIL import Image

# Load image and turn into grayscale
image = cv2.imread('dataset_1/old/image_00001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Adjust image size
image_width, image_height = image.shape
min_side = min(image_width, image_height)
left = (image_width - min_side) // 2
top = (image_height - min_side) // 2
right = left + min_side
bottom = top + min_side
image = image[top:bottom, left:right]
pil_image = Image.fromarray(image)
image = pil_image.resize((512, 512), Image.LANCZOS)

# Extract HOG feature
fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
fd = fd.reshape(1, -1)

# Load model
model_filename = "svm_model.joblib"
loaded_model = joblib.load(model_filename)

# Predict
prediction = loaded_model.predict(fd)

print(prediction)