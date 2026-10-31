import os
env_path = r"C:\Users\LENOVO\anaconda3\envs\tf210-gpu-env"
os.environ['PATH'] = env_path + r"\bin;" + env_path + r"\Library\bin;" + os.environ['PATH']

import tensorflow as tf
print(tf.__version__)

import matplotlib
matplotlib.use('Agg')

import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral

import pandas as pd
import numpy as np
import time
import seaborn as sn

# init_notebook_mode(connected=True)
# %matplotlib inline

## GLOBAL VARIABLES
dataset = 'IP'
test_ratio = 0.9
windowSize = 11

def loadData(name):
    import sys as _sys_ld, os as _os_ld, scipy.io as _sio_ld
    _sys_ld.path.insert(0, '/kaggle/working')
    try:
        from common_data import get_hsi_data_dir
        data_path = get_hsi_data_dir()
    except Exception:
        data_path = '/kaggle/input/dataset-mp'
    if name == 'IP':
        data   = _sio_ld.loadmat(_os_ld.path.join(data_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = _sio_ld.loadmat(_os_ld.path.join(data_path,'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data   = _sio_ld.loadmat(_os_ld.path.join(data_path,'Salinas_corrected.mat'))['salinas_corrected']
        labels = _sio_ld.loadmat(_os_ld.path.join(data_path,'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data   = _sio_ld.loadmat(_os_ld.path.join(data_path,'PaviaU.mat'))['paviaU']
        labels = _sio_ld.loadmat(_os_ld.path.join(data_path,'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data   = _sio_ld.loadmat(_os_ld.path.join(data_path,'KSC.mat'))['KSC']
        labels = _sio_ld.loadmat(_os_ld.path.join(data_path,'KSC_gt.mat'))['KSC_gt']
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return data, labels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from common_data import get_hsi_data_dir
    data_path = get_hsi_data_dir()
except Exception:
    data_path = r"C:\Users\LENOVO\Pictures\MinorProject\data"
    
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt'] 
    
    return data, labels

import pandas as pd
import numpy as np

def extract_pixels(X, y):
  q = X.reshape(-1, X.shape[2])
  df = pd.DataFrame(data = q)
  df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
  df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]+['class']

  return df

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

from skimage.morphology import reconstruction
from sklearn.preprocessing import QuantileTransformer

def CRV(X, band_no, w, h, numComponents):

  X_mean = X
  X_mean['mean'] = X_mean.mean(axis=1)
  Avr = [X_mean['mean']]
  rep_Avr = np.repeat(Avr, band_no)
  rep_Avr =np.reshape(rep_Avr, (-1,band_no))
  main_X = X.iloc[:, :-1]
  seed = main_X - rep_Avr
  dilated = reconstruction(seed.values, main_X.values)
  hdome = main_X - dilated


  hdome_raw_mean = hdome
  hdome_raw_mean = pd.concat([hdome_raw_mean, hdome_raw_mean.agg(['mean'], axis=0)])
  sort_hdome_raw_mean = hdome_raw_mean.sort_values(by='mean', axis=1, ascending=False)
  Top_columns = sort_hdome_raw_mean[sort_hdome_raw_mean.columns[0:numComponents]]
  columns_Names = Top_columns.columns

  input_data = df[columns_Names]

  scaler = QuantileTransformer(n_quantiles=15, random_state=0,output_distribution='normal') #
  scaler = scaler.fit_transform(input_data)
  scaler = np.reshape(scaler, (w, h, numComponents))

  return scaler


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

X1, y1 = loadData(dataset)
band_no = X1.shape[2]
h = X1.shape[1]
w = X1.shape[0]
K = 15  # The target number of bands
X1.shape, y1.shape, h, w, band_no

df = extract_pixels(X1, y1)
X = df.iloc[:, :-1]

CRV_X = CRV(X, band_no, w, h, numComponents= K)
X = CRV_X
X.shape

X, y = createImageCubes(X, y1, windowSize=windowSize)

X.shape, y.shape


Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape


Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)

Xtrain.shape

ytrain = np_utils.to_categorical(ytrain)

ytrain.shape

S = windowSize
L = K
output_units = 9 if (dataset == 'PU' or dataset == 'PC') else (16 if dataset == 'IP' else 13)

## input layer
input_layer = Input((S, S, L, 1))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
print(conv_layer3.shape)
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

flatten_layer = Flatten()(conv_layer4)

## fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = BatchNormalization()(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = BatchNormalization()(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()

# compiling the model
adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# checkpoint
filepath = "best-model_CRV_MHDL.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

import time
training_start_time = time.time()
history = model.fit(x=Xtrain, y=ytrain, batch_size=32, epochs=100, callbacks=callbacks_list)
training_end_time = time.time()

print("Training time : ", training_end_time - training_start_time )
model1_Training_time = training_end_time - training_start_time

import os
path = 'models'
if not os.path.exists(path):
    os.makedirs(path)

# save model
filename = 'model_1.h5'
model.save(path +'/'+filename)
print('>Saved %s' % filename)

print(history.history.keys())

ac = history.history['accuracy']
accuracy_report = pd.DataFrame(ac)
accuracy_report.to_csv('Model_accuracy.csv', index= False)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('model_accuracy.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss'], loc='upper right')
plt.savefig('model_loss.png')
plt.close()

ls = history.history['loss']
accuracy_report = pd.DataFrame(ls)
accuracy_report.to_csv('Model_loss.csv', index= False)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
Xtest.shape

ytest = np_utils.to_categorical(ytest)
ytest.shape

Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)

classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (X_test,y_test,name):
    start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    end = time.time()

    TESTING_TIME = end - start
    print(TESTING_TIME)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'KSC':
        target_names = ["Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
    
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    ###################################
    clsf_report = pd.DataFrame(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names, output_dict=True)).transpose()
    clsf_report.to_csv(dataset+'_Model Classification Report.csv', index= True)
    #########################
    df_cm = pd.DataFrame(confusion, columns=target_names, index = target_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,8))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
    plt.savefig('confusion_matrix_model.png', dpi=300)
    ##############
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    return TESTING_TIME, classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100

TESTING_TIME, classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
df_clas = classification
classification = str(classification)
confusion = str(confusion)
element_mean = np.mean(each_acc, axis=0)
element_std = np.std(each_acc, axis=0)
file_name = "classification_Results.txt"

with open(file_name, 'w') as x_file:
    x_file.write('{} Training Time (%)'.format(model1_Training_time))
    x_file.write('\n')
    x_file.write('{} Test Time (%)'.format(TESTING_TIME))
    x_file.write('\n')
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('{}+{} mean_KAPPA ± std_KAPPA is:'.format(str(np.mean(kappa)), str(np.std(kappa))))
    x_file.write('\n')
    x_file.write('{}+{} mean_OA ± std_OA is: (%)'.format(str(np.mean(oa)), str(np.std(oa))))
    x_file.write('\n')
    x_file.write('{}+{} mean_AA ± std_AA is:'.format(str(np.mean(aa)), str(np.std(aa))))
    x_file.write('\n')
    x_file.write('{} AMean of all elements in confusion matrix: (%)'.format(str(element_mean)))
    x_file.write('\n')
    x_file.write('{} Standard deviation of all elements in confusion matrix: (%)'.format(str(element_std)))
    x_file.write('\n')
    x_file.write('==============================\n')
    x_file.write('{}classification Report:'.format(classification))
    x_file.write('\n')
    x_file.write('===============================\n')
    x_file.write('{} confusion Matrix:'.format(confusion))

def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

height = y1.shape[0]
width = y1.shape[1]
PATCH_SIZE = windowSize
numComponents = K

X = padWithZeros(CRV_X, PATCH_SIZE//2)

# Vectorized predicted output map
outputs1 = np.zeros((height, width))

coords = []
patches = []
for i in range(height):
    for j in range(width):
        target = int(y1[i, j])
        if target != 0:
            coords.append((i, j))
            image_patch = Patch(X, i, j)
            patches.append(image_patch)

if len(patches) > 0:
    patches = np.array(patches)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, numComponents, 1).astype('float32')
    
    # Run prediction in batches of 64
    pred_list = []
    batch_size_pred = 64
    for b_idx in range(0, len(patches), batch_size_pred):
        batch = patches[b_idx : b_idx + batch_size_pred]
        preds = model.predict(batch, verbose=0)
        preds = np.argmax(preds, axis=1)
        pred_list.extend(preds.tolist())
        
    for idx, (i, j) in enumerate(coords):
        outputs1[i][j] = pred_list[idx] + 1

plt.figure(figsize=(7,7))
plt.imshow(y1)
plt.title("Ground Truth")
plt.axis('off')
plt.savefig('ground_truth_IP_model.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(7,7))
plt.imshow(outputs1.astype(int))
plt.title("Classification Prediction Map (CRV)")
plt.axis('off')
plt.savefig('predictions_IP_model.png', bbox_inches='tight')
plt.close()

print("Prediction maps ground_truth_IP_model.png and predictions_IP_model.png saved successfully!") 

