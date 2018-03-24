# Convolution Neural Network using Keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

IMAGE_SIZE = 32
EPOCHS = 1

# If the input size is increased, it will require more computing power
classifier.add(Conv2D(32, (3, 3), input_shape=(
    IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

# Using units more than 100 gives better results. Convention is to use a power of 2
classifier.add(Dense(activation='relu', units=128))
# Activation = "sigmoid" for binary if more than two then use softmax
classifier.add(Dense(activation='sigmoid', units=1))

# Loss = "binary_crossentropy" for binary if more than two then use "categorical_crossentropy"
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    '../dataset/training_set',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    '../dataset/test_set',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=EPOCHS,
    validation_data=test_set,
    validation_steps=2000)

# Saving model
classifier.save("model.h5")
