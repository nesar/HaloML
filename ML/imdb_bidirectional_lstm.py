'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


max_features = 1000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32
epochs=10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])
          
y_pred = model.predict(x_test)

plotLossAcc = False
if plotLossAcc:
    import matplotlib.pylab as plt

    train_loss= model.fit.history['loss']
    val_loss= model.fit.history['val_loss']
    train_acc= model.fit.history['acc']
    val_acc= model.fit.history['val_acc']
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

    train_loss= model.fit.history['loss']
    val_loss= model.fit.history['val_loss']
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
    
    diff = np.abs(y_pred.T - y_test)

    print('Difference min, max, mean, std, median')
    print(np.min(diff), np.max(diff), np.mean(diff), np.std(diff), np.median(diff))

    fig, ax = plt.subplots(2,1, figsize = (4,10))
    fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace= 0.3)
    ax[0].scatter(y_pred.T, y_test)
    ax[0].set_ylabel('y_test[0] --- m200')
    ax[0].set_xlabel('y_pred[0]')


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


 

