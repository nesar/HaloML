'''
Trains and evaluate a simple MLP
based on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

#import json
m_particle = 3.65235543e10
m_particle = 1.3e8
max_words = 30 # 1000

batch_size = 32
epochs = 10
l_r = 0.1
d_r = 0.0

print('Loading data...')

# x_train = density profile.

class density_profile:
    def __init__(self, data_path, para_path1, para_path2, test_split = 0.2, num_para = 1):
        self.data_path = data_path
        self.para_path1 = para_path1
        self.para_path2 = para_path2
        self.num_para = num_para
        self.test_split = test_split


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
        allData = allData[shuffleOrder]/(1e4*m_particle)
        allPara1 = allPara1[shuffleOrder]/(1e4*m_particle)
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



print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')



# ------------------------------------------

print('Building model...')
model = Sequential()
model.add(Dense(20, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('linear'))

adam = Adam(lr = l_r, decay = d_r)

model.compile(loss='mean_squared_error', optimizer=adam)


ModelFit = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)


print('---------------------------')

#--------------Testing----------------------------

y_pred = model.predict(x_test)

#-------------------------------------------------

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

#----------- Plot prediction vs truth ----------- 
ScatterPredReal = True
if ScatterPredReal:
    
    diff = np.abs(y_pred[:,0] - y_test[:,0])

    print('Difference min, max, mean, std, median')
    print(np.min(diff), np.max(diff), np.mean(diff), np.std(diff), np.median(diff))

    fig, ax = plt.subplots(2,1, figsize = (4,10))
    fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace= 0.3)
    ax[0].scatter(y_pred[:,0], y_test[:,0])
    ax[0].set_ylabel('y_test[0] --- m200')
    ax[0].set_xlabel('y_pred[0]')

    ax[1].scatter(y_pred[:,1], y_test[:,1])
    ax[1].set_ylabel('y_test[1] --- r200')
    ax[1].set_xlabel('y_pred[1]')

    plt.show()


ScatterPred= True
if ScatterPred:
    fig, ax = plt.subplots(2,1, figsize = (4,10))
    fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace= 0.3)
    ax[0].scatter(y_test[:,0], y_pred[:,0]/y_test[:,0])
    ax[0].set_ylabel('pred/test --- m200')
    ax[0].set_xlabel('y_test[0]')
    #ax[0].set_ylim(0,2)

    ax[1].scatter(y_test[:,0], y_pred[:,1]/y_test[:,1])
    ax[1].set_ylabel('pred/test --- r200')
    ax[1].set_xlabel('y_test[1] --- m200')
    ax[0].set_title('pred/test ratio variation with mass and radius')

    plt.show()


JointDistribution = True
if JointDistribution:
    fig, ax = plt.subplots(1,1, figsize = (5,5))
    fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace= 0.3)
    ax.scatter(y_test[:,0], y_test[:,1], label = 'y_test')
    ax.set_ylabel('m200')
    ax.set_xlabel('r200')
    # ax.set_title()

    ax.scatter(y_pred[:,0], y_pred[:,1], label = 'y_pred')
    ax.set_ylabel('m200')
    ax.set_xlabel('r200')
    ax.set_title('Joint distribution')
    plt.legend()

    plt.show()


 
