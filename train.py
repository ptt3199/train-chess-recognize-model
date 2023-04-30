import cv2
import os
import numpy as np
from keras.utils.image_utils import img_to_array
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras import backend as k
from keras.callbacks import EarlyStopping

# homemade
from load_data import document_chess_info
import g_params

train_image_path = g_params.train_image_path
all_chess_data_path = document_chess_info(train_image_path)
epochs = 30
test_ratio = 5
count_image = 0
size = g_params.image_train_size
classes = 15

data = []
label = []
test_data = []
test_label = []
for each_image in all_chess_data_path:
    count_image += 1
    print(each_image)
    image = cv2.imread(each_image, cv2.COLOR_BGR2GRAY)
    image.resize((size, size))
    image_array = img_to_array(image)
    image_label = g_params.chess_table.index(each_image.split(os.path.sep)[-2])
    if count_image % test_ratio == 0:
        test_data.append(image)
        test_label.append(image_label)
    else:
        data.append(image)
        label.append(image_label)

# create train data
data = np.array(data)
data = data/255.0
print("data shape =", data.shape, "===============================")
label = np.array(label)
label = to_categorical(label, num_classes=classes)

index = np.arange(data.shape[0])
np.random.shuffle(index)
data = data[index]
label = label[index, :]

# create test data
test_data = np.array(test_data)
test_data = test_data/255.0
print("test_data shape =", test_data.shape, "===============================")
test_label = np.array(test_label)
test_label = to_categorical(test_label, num_classes=classes)

test_index = np.arange(test_data.shape[0])
np.random.shuffle(test_index)
test_data = test_data[test_index]
test_label = test_label[test_index, :]

print("Data loaded.")

# train
x_train = data
y_train = label
x_test = test_data
y_test = test_label

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                 input_shape=(size, size, 1), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                 input_shape=(size, size, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                 input_shape=(size, size, 1), padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                 input_shape=(size, size, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                 input_shape=(size, size, 1), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                 input_shape=(size, size, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(15, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=4, mode='max')
print("default learning rate =", model.optimizer.learning_rate)
k.set_value(model.optimizer.learning_rate, 0.0001)
print("current learning rate =", model.optimizer.learning_rate)
history = model.fit(x_train, y_train, batch_size=36, epochs=epochs, callbacks=[early_stopping], verbose=1,
                    validation_split=0.2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

data_size = len(y_train)
print("train len  = ", data_size)
print("x_test size = ", len(x_test))
print("y_test size = ", len(y_test))

score = model.evaluate(x_test, y_test)
print('acc', score[1])

# saving the model
model_path = g_params.model_path
model_name = "model_moi.h5"
model.save(os.path.join(model_path, model_name))

if not os.path.exists(model_path):
    os.makedirs(model_path)

print('Saved trained model at %s ' % model_path)
print("Train model done!")
print("Model saved.")
