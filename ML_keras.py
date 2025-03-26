# Machine learning module for Python
# ==================================
#
# Python module for segmenting fracture surface to 3 categories (background, brittle and ductile) using Keras
# ==================================

from keras import layers
from sklearn.utils.class_weight import compute_class_weight
import os
import math
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Conv1D, RandomFlip
import numpy as np
from image_ML2 import make_data_split, array_crop,array_crop2, array_fft, eval_sfa
import time
import cv2
import matplotlib.pyplot as plt

start_time = time.time()
print('test')
def train_conv_nn(train=True):
    start_train_time = time.time()
    X_train, X_test, y_train, y_test= make_data_split()
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    shape_length =int(math.sqrt(X_train.shape[1]))
    X_train = X_train.reshape(X_train.shape[0],1, shape_length, shape_length)
    X_test = X_test.reshape(X_test.shape[0], 1,shape_length, shape_length)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = to_categorical(y_train, 3)
    Y_test = to_categorical(y_test, 3)

    # AUGMENTATION
    print('START AUGMENTATION')
    data_augmentation = Sequential([
        # layers.RandomBrightness(factor=(0.2, 0.7), value_range=(0, 1)),
        layers.RandomRotation(0.2),
        # layers.RandomContrast(factor=(0.2), value_range=(0, 1))
    ])
    data_augmentation_vis = Sequential([
        # layers.RandomBrightness(factor=(0.2, 0.7), value_range=(0, 1)),
        layers.RandomRotation([0.2,0.5]),
        # layers.RandomContrast(factor=(0.2), value_range=(0, 1))

    ])
    ## VISUALIZE AUGMENTATION
    image_vis = next(iter(X_train))
    image_vis = image_vis.reshape(image_vis.shape[1], image_vis.shape[1], 1)
    _ = plt.imshow(image_vis)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_image = data_augmentation_vis(image_vis)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image,cmap='gray')
        plt.axis("off")

    result_vis = (image_vis)
    _ = plt.imshow(result_vis)
    plt.savefig(
        'visualize_aug' + '.png',
        dpi=200)
    plt.clf()
    ## END OF VISUALIZE AUGMENTATION

    model = Sequential([])
    # model = Sequential([data_augmentation])

    model.add(Conv2D(32,(6, 6), activation = 'relu', input_shape=(1,shape_length,shape_length), data_format='channels_first'))
    model.add(Conv2D(32, (6, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(3, activation='softmax'))

    # WEIGHTS
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.tolist()), y=y_train.tolist())
    print('class_weights:', class_weights)
    weight_bg = class_weights[0]
    weight_brittle = class_weights[1]
    weight_ductile = class_weights[2]

    class_weights = {
        0: weight_bg,
        1: weight_brittle,
        2: weight_ductile,
    }
    # END OF WEIGHTS

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('(Y_train)', type(Y_train), (Y_train.shape))
    # print('(Y_train)', Y_train)
    print('(X_train)', type(X_train), (X_train.shape))
    history =model.fit(X_train, Y_train,
              batch_size=64, epochs=3, verbose=1,class_weight=class_weights) # working

    score = model.evaluate(X_test, Y_test, verbose=1)
    print('score:',score)
    y_prediction = model.predict(X_test)
    y_prediction = np.argmax(y_prediction, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    result = confusion_matrix(Y_test, y_prediction, normalize='pred')
    print('cm:',result)
    model.save('keras_clf_model.keras')

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    # plt.show()

    y_pred = model.predict(X_test)  # y_pred - np.array
    y_pred_single = np.asarray(([int(max(sublist)) for sublist in y_pred]))
    cm = confusion_matrix(y_test, y_pred_single )
    fig = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig.plot()
    fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
    plt.savefig('cmx.png')
    plt.clf()

    print("--- training duration %s seconds ---" % (time.time() - start_train_time))
    print('*** start testing ***')

def eval_sfa(mlp_coefs = 'img_mlp_clf_params.pkl',train_nn = True):
    scale_x = scale_y = 15
    global model
    images_folder = r'ML_image/fracture_images/'
    start_time = time.time()

    k=0
    for img in os.listdir(images_folder):
        print(img)
        k+=1
        img_path = os.path.join(images_folder, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_width, img_height = img.shape[:2]
        img_pad = np.abs((img_width - img_height) / 2)  # half of side pad (2 sides)
        img = img[0:img_height, int(img_pad):int(1920 + img_pad)]
        img = cv2.resize(img, dsize=(1920, 1920), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img)
        print('shape', img.shape)
        if img.shape != (1920, 1920):
            raise Exception('wrong resolution of image '+str(img) )
    print(str(k)+' images resolution ok')

    ## LOADING COEFS
    loaded_model = load_model('keras_clf_model.keras')
    selected_clf = loaded_model
    ## END LOADING COEFS

    print('*** start testing ***')
    for img_name in os.listdir(images_folder):
        split_image = []
        k+=1
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tot_subimages = 0
        y_len,x_len=img.shape
        array_crop2(img=img, main_array=split_image, fft_transform=False)
        split_image=np.asarray(split_image)
        b = np.array(255)
        split_image = split_image/b
        split_image_shape_length = int(math.sqrt(split_image.shape[1]))
        split_image = split_image.reshape(split_image.shape[0], 1, split_image_shape_length, split_image_shape_length)
        y_pred_img = selected_clf.predict(split_image)  # y_pred - np.array
        y_pred_img = np.around(y_pred_img)
        plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation="nearest")
        plt.title(str(img_name))
        y_pred_img = np.argmax(y_pred_img, axis=1)
        unique, counts = np.unique(y_pred_img, return_counts=True)
        f=dict(zip(unique, counts))
        try:
            brittle_patches = f[1]
            ductile_patches = f[2]
            if brittle_patches + ductile_patches != 0:
                eval_sfa_by_nn = int(ductile_patches / (brittle_patches + ductile_patches) * 100)
        except: eval_sfa_by_nn =brittle_patches=0

        # Plot the mask
        k=0
        my_dict = {}
        color_dict = {0: 'blue' , 1:'red', 2:'green'}
        for y in range(scale_y):
            for x in range(scale_x):
                try:
                    rect_index=y_pred_img[k]
                    k+=1
                    tot_subimages+=1
                    x_coords = int(((x+0.5)*x_len)/scale_x)
                    y_coords = int(((y+0.5)*y_len)/scale_y)
                    my_dict[(x, y)] = rect_index
                    rect = Rectangle((x_coords, y_coords), 128, 128, color=color_dict[rect_index], fill=True, alpha = 0.2, ec=None)
                    plt.gca().add_patch(rect)
                except:
                    pass
        plt.savefig(
            r'ML_image/measured_fracture_images/'+str(img_name))
        plt.clf()
        print("eval duration %s seconds ---" % (time.time() - start_time))

print('start training')
train_conv_nn()
print('start evaluating')
eval_sfa()
