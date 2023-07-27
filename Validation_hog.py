from skimage import feature, exposure
import cv2
import joblib
from PIL import Image

def valid(no):
    # Load image and turn into grayscale
    no = str(no)
    no = no.zfill(5)
    print(no)
    image = cv2.imread('dataset_1/valid/image_'+no+'.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG feature
    fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    fd = fd.reshape(1, -1)

    # Load model
    model_filename = "hog_old_svm_model.joblib"
    loaded_model = joblib.load(model_filename)

    # Predict
    prediction = loaded_model.predict(fd)

    print(prediction)
    
for i in range(1,5):
    valid(i)