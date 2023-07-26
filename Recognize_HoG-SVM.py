from skimage import feature, exposure
import cv2
import joblib
from PIL import Image

# 加载识别图像
image = cv2.imread('C:/Users/86561/Desktop/picture_test/shanghai.jpg')
# 将图像转换为灰度图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", image)

# 等待按下任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像预处理
image_width, image_height = image.shape
min_side = min(image_width, image_height)
left = (image_width - min_side) // 2
top = (image_height - min_side) // 2
right = left + min_side
bottom = top + min_side
image = image[top:bottom, left:right]

# 将NumPy数组转换为PIL图像对象
pil_image = Image.fromarray(image)

# 调整图像大小
image = pil_image.resize((512, 512), Image.LANCZOS)

# 使用HOG提取特征
fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

# 将特征向量转换为2维数组
fd = fd.reshape(1, -1)

# 加载训练好的模型
model_filename = "svm_model.joblib"
loaded_model = joblib.load(model_filename)

# 执行预测
prediction = loaded_model.predict(fd)

print(prediction)