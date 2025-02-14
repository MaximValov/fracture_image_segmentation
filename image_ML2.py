import matplotlib.pyplot as plt
import os
import pickle
import math
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,mean_squared_log_error,log_loss
from sklearn import datasets, metrics, svm
import cv2
import numpy as np
from sklearn.utils.validation import check_consistent_length,check_array
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import GridSearchCV
import time
from functools import reduce


scale_x =scale_y= 15

def array_fft(img_path):
    # Compute the discrete Fourier Transform of the image
    fourier = cv2.dft(np.float32(img_path), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)
    # calculate the magnitude of the Fourier Transform
    magnitude = 20 * np.log(cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))
    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    scale_mag = 2 # we take 1/4 of fft image due to quarter - symmetry
    y_mag_len, x_mag_len = magnitude.shape
    cropped_image = magnitude[0:int((y_mag_len) / scale_mag),
                    0:int((x_mag_len) / scale_mag)]
    return cropped_image

def array_crop(img, main_array, fft_transform=False):
    global tot_no_of_blocks, no_of_blocks, selected_clf, scale_y, scale_x
    tot_no_of_blocks =0
    no_of_blocks =0
    y_len, x_len = img.shape
    for y in range(scale_y):
        for x in range(scale_x):
            tot_no_of_blocks += 1
            cropped_image = img[int((y * y_len) / scale_y):int(((y + 1) * y_len) / scale_y),
                            int((x * x_len) / scale_x):int(((x + 1) * x_len) / scale_x)]
            if fft_transform:
                cropped_image = array_fft(cropped_image)
            if len(np.unique(cropped_image)) > 1:  # nonempty subimages
                count_zeros = len([element for element in cropped_image.flatten() if element == 0])
                if fft_transform:
                    count_zeros = 0
                if count_zeros == 0:  # take only images wo empty pixels
                    main_array.append(cropped_image.flatten())
                    no_of_blocks += 1

def array_crop2(img, main_array, fft_transform=False):
    global tot_no_of_blocks, no_of_blocks, selected_clf, scale_y, scale_x
    tot_no_of_blocks =0
    no_of_blocks =0
    y_len, x_len = img.shape
    for y in range(scale_y):
        for x in range(scale_x):
            tot_no_of_blocks += 1
            cropped_image = img[int((y * y_len) / scale_y):int(((y + 1) * y_len) / scale_y),
                            int((x * x_len) / scale_x):int(((x + 1) * x_len) / scale_x)]
            if fft_transform:
                cropped_image = array_fft(cropped_image)
            main_array.append(cropped_image.flatten())
            no_of_blocks += 1

def make_data_split():
    global mlp_coefs, tot_no_of_blocks,no_of_blocks

    brittle_2Darray=[]
    ductile_2Darray=[]
    bg_2Darray=[]
    split_image_brittle=[]
    split_image_ductile=[]
    split_image_bg=[]
    spl=[split_image_brittle,split_image_ductile,split_image_bg]
    folder_paths = [r'ML_image/brittle_train_images/',r'ML_image/ductile_train_images/',r'ML_image/bg_train_images/']
    pattern = r'\.(jpg|png)$'
    for s,f in zip(spl, folder_paths):
        i=0
        for file in os.listdir(f):
            if re.search(pattern, file):
                file_path = os.path.join(f, file)
                if os.path.isfile(file_path):
                    s.append(file_path)
                    i+=1

    split_images=[brittle_2Darray, ductile_2Darray, bg_2Darray]
    subimage_arrays = []
    no_of_blocks = 0
    tot_no_of_blocks = 0
    for img in (split_image_brittle):
        folder= re.findall(r'\b35\w*', img)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(1920, 1920), interpolation=cv2.INTER_CUBIC)
        if img.shape != (1920, 1920):
            raise Exception('wrong resolution of image '+str(img) )
        array_crop(img=img, main_array=brittle_2Darray)
    no_of_blocks_brittle = no_of_blocks
    no_of_blocks = 0
    tot_no_of_blocks = 0
    for img in (split_image_ductile):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(1920, 1920), interpolation=cv2.INTER_CUBIC)
        if img.shape != (1920, 1920):
            raise Exception('wrong resolution of image '+str(img) )
        array_crop(img=img,main_array=ductile_2Darray)
    no_of_blocks_ductile = no_of_blocks
    no_of_blocks = 0
    tot_no_of_blocks = 0
    for img in (split_image_bg):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(1920, 1920), interpolation=cv2.INTER_CUBIC)
        if img.shape != (1920, 1920):
            raise Exception('wrong resolution of image '+str(img) )
        array_crop(img=img, main_array=bg_2Darray)
    no_of_blocks_bg = no_of_blocks
    check_array(split_images[0],ensure_2d=True)
    check_array(split_images[1],ensure_2d=True)
    check_array(split_images[2],ensure_2d=True)

    print('***LEARNING***')
    subimage_arrays.extend(brittle_2Darray)
    subimage_arrays.extend(ductile_2Darray)
    subimage_arrays.extend(bg_2Darray)
    labels = []
    labels1 = np.array([1]*len(split_images[0])) # Replace with corresponding label data (0-9)
    labels2 = np.array([2]*len(split_images[1])) # Replace with corresponding label data (0-9)
    labels3 = np.array([0]*len(split_images[2])) # Replace with corresponding label data (0-9)
    labels.extend(labels1)
    labels.extend(labels2)
    labels.extend(labels3)

    X_train, X_test, y_train, y_test = train_test_split(subimage_arrays, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)
    return X_train, X_test, y_train, y_test

def make_nn(mlp_coefs='img_mlp_clf_params.pkl',partial_fit =True,loss_curve_show = False,solver_type ='sgd', tuning =False):
    X_train, X_test, y_train, y_test = make_data_split()
    mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 50, 30), learning_rate='constant',
                                max_iter = 500,activation = 'tanh', alpha=0.0001,
                                solver = solver_type,verbose=0) #hyp

    if mlp_coefs == '':
        if partial_fit == True:
            # Обучение модели
            history = mlp_clf.fit(X_train, y_train)
            # Вывод результатов
            print("Loss:", history.loss_)
            print("Accuracy:", history.score(X_train, y_train))
        else:
            mlp_clf.fit(X_train, y_train)  # расчет коэфф модели по тренир. данным
            with open('img_mlp_clf_params.pkl', 'wb') as f:
                pickle.dump(mlp_clf, f, protocol=0)
                print('dumped')

        mlp_coefs='img_mlp_clf_params.pkl'
        with open(mlp_coefs, 'rb') as f:  # загрузка коэфф модели
            mlp_clf = pickle.load(f)

    selected_clf= mlp_clf
    if mlp_coefs=='':
        y_pred = selected_clf.predict(X_test)  # y_pred - np.array
        y_pred_train = selected_clf.predict(X_train)
        print("Training set score(R2): %f" % selected_clf.score(X_train, y_train), 'samples:', len(X_train))
        print("Testing set score(R2): %f" % selected_clf.score(X_test, y_test), 'samples:', len(X_test))
        print('y_pred',( y_pred[0]),)
        cm = confusion_matrix(y_test, y_pred, labels=selected_clf.classes_, )
        fig = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=selected_clf.classes_)
        fig.plot()
        fig.figure_.suptitle("Confusion Matrix")
        plt.show()


    # # hyper parameter tunung
    if tuning :
        print('start param tuning')
        start_time = time.time()
        param_grid = {
            'hidden_layer_sizes': [(150,50), ],
            'max_iter': [500,2000],
            'activation': ['tanh', 'relu','identity'],
            'solver': ['sgd', 'adam','lbfgs'],
            'alpha': [0.0001, 0.00001],
            'learning_rate': ['constant','adaptive','invscaling'],
        }
        grid = GridSearchCV(selected_clf, param_grid, n_jobs= -1, cv=5)
        grid.fit(X_train, y_train)

        print('grid',grid.best_params_, type(grid.best_params_))
        print("--- tuning: %s seconds ---" % (time.time() - start_time))
    # # end of hyper parameter tunung

    if loss_curve_show == True:
        if solver_type != 'lbfgs':
            mse_train = root_mean_squared_error([y_train], [y_pred_train])
            mse_test = root_mean_squared_error(y_test, y_pred)
            mae_train = mean_squared_log_error([y_train], [y_pred_train])
            mae_test = mean_squared_log_error(y_test, y_pred)
            print('mse_train',mse_train)
            print('mse_test',mse_test)
            print('mae_train',mae_train)
            print('mae_test',mae_test)
            plt.plot(selected_clf.loss_curve_, label="train")
            plt.title("Loss Curve", fontsize=14)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.legend()
            plt.show()

def eval_sfa(mlp_coefs = 'img_mlp_clf_params.pkl'):
    images_folder = r'ML_image/fracture_images/'
    start_time = time.time()
    ## CHECKING IMAGES RESOLUTION##
    k=0
    for img in os.listdir(images_folder):
        k+=1
        img_path = os.path.join(images_folder, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_width, img_height = img.shape[:2]
        img_pad = (img_width - img_height) / 2  # half of side pad (2 sides)
        img = img[0:img_height, int(img_pad):int(1920 + img_pad)]
        img = cv2.resize(img, dsize=(1920, 1920), interpolation=cv2.INTER_CUBIC)
        plt.imshow(np.array(img), cmap=plt.cm.gray, interpolation="nearest")
        ax = plt.axes()
        ax.set_axis_off()
        plt.savefig(img_path)
        plt.clf()

        if img.shape != (1920, 1920):
            raise Exception('wrong resolution of image '+str(img) )
    ## END CHECKING IMAGES RESOLUTION##
    ## LOADING COEFS
    with open(mlp_coefs, 'rb') as f:  # загрузка коэфф модели
        mlp_clf = pickle.load(f)
    ## END LOADING COEFS
    selected_clf= mlp_clf
    print('*** start testing ***')
    for img_name in os.listdir(images_folder):
        split_image = []
        k+=1
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation="nearest")
        plt.show()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tot_subimages = 0
        y_len,x_len=img.shape
        no_of_blocks=0
        array_crop(img=img, main_array=split_image,predict=False)
        y_pred_img = selected_clf.predict(split_image)  # y_pred - np.array
        plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation="nearest")
        plt.title(str(img_name))
        unique, counts = np.unique(y_pred_img, return_counts=True)
        f=dict(zip(unique, counts))
        brittle_patches = np.count_nonzero(y_pred_img == 1)
        ductile_patches = np.count_nonzero(y_pred_img == 2)
        eval_sfa_by_nn = int(ductile_patches/(brittle_patches+ductile_patches)*100)
        print(img_name)
        print('eval_sfa_by_nn = ',eval_sfa_by_nn)
        # Plot the mask
        k=0
        my_dict = {}
        color_dict = {0: 'blue' , 1:'red', 2:'green'}
        print('len y_pred_img = ', type(y_pred_img), (y_pred_img.shape))
        print('y_pred_img = ', y_pred_img)
        for y in range(scale_y):
            for x in range(scale_x):
                try:
                    rect_index=y_pred_img[k]
                    k+=1
                    tot_subimages+=1
                    x_coords = int(((x+0.5)*x_len)/scale_x)
                    y_coords = int(((y+0.5)*y_len)/scale_y)
                    my_dict[(x, y)] = rect_index
                    plt.scatter(x_coords, y_coords, marker='o', color=color_dict[rect_index])
                except:
                    pass
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.savefig(
            r'ML_image/measured_fracture_images/'+str(img_name))
        plt.show()


def main():
    print('-start_ML2-')
    make_nn(mlp_coefs='', partial_fit=False, solver_type='lbfgs')
    eval_sfa()
    print('-end_ML2-')

if __name__ == "__main__":
    main()
