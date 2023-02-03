from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model('model.h5')
class_names=["YES","NO"]
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open('test.jpg').convert('RGB')
image = ImageOps.fit(image,(224, 224), Image.ANTIALIAS)
image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

prediction = model.predict(data)

index = np.argmax(prediction)
class_name = class_names[index]

print("Tình trạng bệnh : ", class_name)
