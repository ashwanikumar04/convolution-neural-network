import numpy as np
from keras.preprocessing import image
from keras.models import load_model
classifier = load_model("model.h5")


test_image = image.load_img('image2.jpg', target_size=(32, 32))
test_image = image.array_to_img(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    print('dog')
else:
    print('cat')
