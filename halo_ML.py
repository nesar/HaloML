'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

import json

m_particle = 3.65e10

max_words = 30 # 1000
batch_size = 32
epochs = 100 # 5

print('Loading data...')

# x_train = density profile.

class density_profile:
    def __init__(self, data_path, para_path1, para_path2, test_split = 0.2, num_para = 1):
        self.data_path = data_path
        self.para_path1 = para_path1
        self.para_path2 = para_path2
        self.num_para = num_para   # Just mass right now
        self.test_split = test_split

        #self.x_train = []
        #self.y_train = []
        #self.x_test = []
        #self.y_test = []

    def open_data(self):                                                        
#        with open(self.data_path) as json_data:
#            self.allData = json.load(json_data)
#
#        with open(self.para_path) as json_data:
#            self.allPara = json.load(json_data)
        
        self.allData = np.load(self.data_path)
        self.allPara1 = np.load(self.para_path1)
        self.allPara2 = np.load(self.para_path2)
        
        return self.allData, self.allPara1, self.allPara2

    def load_data(self): # randomize and split into train and test data

        allData, allPara1, allPara2 = self.open_data()
        num_files = len(allData)                                                
        num_train = int((1-self.test_split)*num_files)

        np.random.seed(1234)
        shuffleOrder = np.arange(num_files)
        np.random.shuffle(shuffleOrder)
        allData = allData[shuffleOrder]/m_particle
        allPara1 = allPara1[shuffleOrder]/m_particle
        allPara2 = allPara2[shuffleOrder]
        allPara = np.dstack((allPara1, allPara2))[0]
        print (allPara.shape)

        self.x_train = allData[0:num_train]
        self.y_train = allPara[0:num_train]

        self.x_test = allData[num_train:num_files]
        self.y_test = allPara[num_train:num_files]

        return (self.x_train, self.y_train), (self.x_test, self.y_test)



density_file = 'Bolshoi_All_density_profile.npy'                                            
halo_para_file1 = 'Bolshoi_mass.npy'
halo_para_file2 = 'Bolshoi_Radius.npy'
dens = density_profile(data_path = density_file, para_path1 = halo_para_file1, para_path2 = halo_para_file2)

(x_train, y_train), (x_test, y_test) = dens.load_data()


#(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)

#x_train: #haloes, #bin   -- data
#y_train: #haloes    -- label


print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

#num_classes = np.max(y_train) + 1
#print(num_classes, 'classes')


# Something is not right ------ all x_train, x_test = 0
#print('Vectorizing sequence data...')
#tokenizer = Tokenizer(num_words=max_words)
#xx_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
#xx_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
#print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

# ------------------------------------------

#
#print('Convert class vector to binary class matrix '
#      '(for use with categorical_crossentropy)')
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#print('y_train shape:', y_train.shape)
#print('y_test shape:', y_test.shape)
#
print('Building model...')
model = Sequential()
model.add(Dense(2, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
#model.add(Activation('softmax'))
model.add(Activation('linear'))

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='mean_squared_error', optimizer='adam')


ModelFit = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

#print('Test score:', score[0])
#print('Test accuracy:', score[1])
print('---------------------------')



plotLossAcc = False
if plotLossAcc:
    import matplotlib.pylab as plt

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    train_acc= ModelFit.history['acc']
    val_acc= ModelFit.history['val_acc']
    epoch_array = range(1, epochs+1)


    fig, ax = plt.subplots(2,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax[0].plot(epoch_array,train_loss)
    ax[0].plot(epoch_array,val_loss)
    ax[0].set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[0].legend(['train_loss','val_loss'])


    ax[1].plot(epoch_array,train_acc)
    ax[1].plot(epoch_array,val_acc)
    ax[1].set_ylabel('acc')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[1].legend(['train_acc','val_acc'])

    plt.show()


plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    epoch_array = range(1, epochs+1)


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epoch_array,train_loss)
    ax.plot(epoch_array,val_loss)
    ax.set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss','val_loss'])

    plt.show()



a = model.predict(x_test)
