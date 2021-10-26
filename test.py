import os
# import glob
import h5py
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import numpy as np


batch_size = 16  #was 16

train_images = 'faceshape/training_set'
test_images = 'faceshape/testing_set'


TrainDatagen = ImageDataGenerator(
        preprocessing_function= preprocess_input,
        horizontal_flip = True,
        width_shift_range = 0.1,
        height_shift_range = 0.1
)

TestDatagen = ImageDataGenerator(
    preprocessing_function= preprocess_input
)

train_data = TrainDatagen.flow_from_directory(
    train_images,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_data = TestDatagen.flow_from_directory(
    test_images,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical'
)

class_names = ['Heart','Oblong','Oval','Round','Square']


from tensorflow.keras.models import load_model
model = load_model('new_face_classifier.h5')
model.load_weights('new_model_weights.h5')

model.evaluate(test_data)


# import timeit
# start_time = timeit.default_timer()

# image_test = Image.open("testfolder/Heart/heart (7).jpg")
# image_test = Image.open("testfolder/Oval/oval (2).jpg")
# image_test = Image.open("testfolder/Round/round (396).jpg")
# image_test = Image.open("testfolder/Square/square (462).jpg")

# np.expand_dims(np.array(image_test.resize((224,224)))[:, :, 0:3], axis=0)
# preprocess_input(np.expand_dims(np.array(image_test.resize((224,224)))[:, :, 0:3], axis=0))

# prediction = model(preprocess_input(np.expand_dims(np.array(image_test.resize((224,224)))[:, :, 0:3], axis=0)))
# class_names[np.argmax(prediction)]


ImageFile.LOAD_TRUNCATED_IMAGES = True
total = 0
count = 0
for root, dirs, files in os.walk('/content/testing_set/Round'):
    if files:
        for f in files:
            if os.path.splitext(f)[1] not in ['.jpg']: continue
            total += 1
            test_image = Image.open(os.path.join(root, f))
            prediction = model(preprocess_input(np.expand_dims(np.array(test_image.resize((224,224)))[:, :, 0:3], axis=0)))
            result = class_names[np.argmax(prediction)]
            # print('Path : {}, Prediction : {}, Result : {}'.format(os.path.join(root, f), prediction, result))
            if result != "Round":
                count += 1
print('\n')
print(count, '/', total, '장')
print(100 - (count/total) * 100, "% 예측률")


# 혼동행렬 #
train_model = model.predict(train_data)

from sklearn.metrics import accuracy_score
accuracy_score(test_data, train_model)
# 0.83333333333333337

# 17% 정도의 틀린 부분 보기
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');
