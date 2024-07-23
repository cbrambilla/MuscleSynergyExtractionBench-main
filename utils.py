import numpy as np
from numpy.linalg import norm
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from keras.layers import Activation
from keras import backend as K

class MyClass():
    def __init__(self, param):
        self.param = param

def normalize_syns(syns,tcoef):
    for i in range(len(syns)): #4
        s = norm(syns[i][:])
        syns[i][:] = syns[i][:]/s
        tcoef[:][i] = tcoef[:][i]*s
    return syns,tcoef

''' def custom_activation(x):
    #return abs((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) #abs(tanh)
    return abs(K.tanh(x))'''


def save_object(obj,filename):
    try:
        with open("%s.pickle" % filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

def r_squared(preds, true):
    mean_matrix = np.repeat(true[0].mean(axis=0), true[0].shape[0], axis=0)
    # mean_matrix = np.transpose(np.split(mean_matrix, true[0].shape[1]))
    mean_matrix = np.transpose(np.split(mean_matrix, true.shape[1]))
    # squared_err = np.sum(np.sum((preds - true[0]) ** 2))
    squared_err = np.sum(np.sum((preds - true) ** 2))
    # denominator = ((true[0] - mean_matrix) ** 2).sum()
    denominator = ((true - mean_matrix) ** 2).sum()
    r2 = 1 - squared_err / denominator
    return r2

def setup_AE(n_components, xtrain, xtest):
    # Parameters
    batch = len(xtrain[0])
    # batch = 128
    lr = 0.001
    dropout = 0
    epochs_nb = 2000
    # opt = keras.optimizers.Adam(learning_rate=lr)
    # opt = keras.optimizers.SGD(learning_rate=lr) #stochastic gradient descent
    opt = keras.optimizers.RMSprop(lr)  # best one
    l_1 = 0.0001
    l_2 = 0.001
    val_split = 0.01
    loss_fct = "mse"  # "binary_crossentropy" or "mse" or "mae"
    # encoder_activation = "sigmoid"  # tanh is not good bc it gives me negative tcoefs
    encoder_activation = "relu"
    bias_reg = 0.001

    # scaler = StandardScaler()
    # for i in range(len(xtrain)):
    #     scaler = scaler.fit(xtrain[i])
    #     xtrain[i] = scaler.transform(xtrain[i])
    # xtest[0] = scaler.transform(xtest[0])

    encoder_input = keras.Input(shape=(900, 16))
    x = encoder_input
    x = keras.layers.BatchNormalization(scale=True)(x)
    encoder_output = keras.layers.Dense(n_components,
                                        activation=encoder_activation,
                                        # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                        # kernel_regularizer=keras.regularizers.L1L2(l1=l_1,l2=l_2),
                                        # bias_regularizer=keras.regularizers.L2(bias_reg),
                                        # kernel_constraint=keras.constraints.NonNeg(),name = "encoder_out"
                                        )(x)
    encoder_output = keras.layers.Dropout(dropout, name="encoder_output")(encoder_output)
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    decoder_input = encoder_output
    decoder_output = keras.layers.Dense(16,
                                        # activation = "relu",
                                        # kernel_regularizer=keras.regularizers.L1L2(l1=0,l2=0),
                                        name="decoder",
                                        kernel_constraint=keras.constraints.NonNeg()
                                        )(decoder_input)  # 16*4=64

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
    # autoencoder.summary()
    # autoencoder.compile(opt,loss="mse")
    autoencoder.compile(opt,
                        loss=loss_fct,
                        metrics=['accuracy'])

    history = autoencoder.fit(xtrain,
                              xtrain,
                              batch_size=batch,
                              epochs=epochs_nb,
                              validation_split=val_split,
                              verbose=0)
    plot_history=False
    if plot_history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
    tcoef = encoder.predict(xtest[0].reshape(-1,900,16))[0]
    weight_paths = autoencoder.get_weight_paths()
    syns = weight_paths['decoder.kernel']
    syns = syns.numpy()
    syns,tcoef = normalize_syns(syns,tcoef)

    return autoencoder, syns, tcoef

def setup_AE_all_planes(n_components, xtrain, xtest):
    # Parameters
    batch = len(xtrain[0])
    # batch = 128
    lr = 0.001
    dropout = 0
    epochs_nb = 2000
    # opt = keras.optimizers.Adam(learning_rate=lr)
    # opt = keras.optimizers.SGD(learning_rate=lr) #stochastic gradient descent
    opt = keras.optimizers.RMSprop(lr)  # best one
    l_1 = 0.0001
    l_2 = 0.001
    val_split = 0.01
    loss_fct = "mse"  # "binary_crossentropy" or "mse" or "mae"
    encoder_activation = "relu"
    #encoder_activation = "tanh"  # tanh is not good bc it gives me negative tcoefs
    bias_reg = 0.1
    

    # scaler = StandardScaler()
    # for i in range(len(xtrain)):
    #     scaler = scaler.fit(xtrain[i])
    #     xtrain[i] = scaler.transform(xtrain[i])
    # xtest[0] = scaler.transform(xtest[0])

    encoder_input = keras.Input(shape=(900, 16))
    x = encoder_input
    x = keras.layers.BatchNormalization(scale=True)(x)
    encoder_output = keras.layers.Dense(n_components,
                                        activation=encoder_activation,
                                        # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                        # kernel_regularizer=keras.regularizers.L1L2(l1=l_1,l2=l_2),
                                        # bias_regularizer=keras.regularizers.L2(bias_reg),
                                        # kernel_constraint=keras.constraints.NonNeg(),name = "encoder_out"
                                        )(x)
    encoder_output = keras.layers.Dropout(dropout, name="encoder_output")(encoder_output)
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    decoder_input = encoder_output
    decoder_output = keras.layers.Dense(16,
                                        # activation = "relu",
                                        # kernel_regularizer=keras.regularizers.L1L2(l1=0,l2=0),
                                        name="decoder",
                                        kernel_constraint=keras.constraints.NonNeg()
                                        )(decoder_input)  

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
    # autoencoder.summary()
    # autoencoder.compile(opt,loss="mse")
    autoencoder.compile(opt,
                        loss=loss_fct,
                        metrics=['accuracy'])

    history = autoencoder.fit(xtrain,
                              xtrain,
                              batch_size=batch,
                              epochs=epochs_nb,
                              validation_split=val_split,
                              verbose=1)
    plot_history=False
    if plot_history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
    syns = []
    tcoef = []
    for i in range(len(xtest)):
        x = xtest[i]
        tcoef_x = encoder.predict(x.reshape(-1,900,16))[0]
        weight_paths = autoencoder.get_weight_paths()
        syns_x = weight_paths['decoder.kernel']
        syns_x = syns_x.numpy()
        syns_x_norm,tcoef_x_norm = normalize_syns(syns_x,tcoef_x)
        syns.append(syns_x_norm)
        tcoef.append(tcoef_x_norm)

    return autoencoder, syns, tcoef

def plot_syns(H,W,label,n_components):
    lab =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    fig, axs = plt.subplots(n_components,1)
    fig.set_size_inches(15, 15)
    w = 1
    cols = ['maroon','red','blue','green','orange','purple','brown','pink','grey','olive','cyan','darkcyan','deeppink','dodgerblue']
    tick = ['ES','TM','IF','LT','MT','UT','DA','DM','DP','PT','TL','TLa','BL','BS','PR','BR']

    for i in range(n_components):
        axs[i].bar(lab, H[i,:], color =cols[i],edgecolor = 'black', width = w,tick_label = tick)
        axs[i].set_xticklabels(tick)

    plt.xlabel("")
    plt.ylabel("")
    plt.suptitle("synergies " + label)
    plt.show()

    fig, ax = plt.subplots(n_components,1,figsize=(8, 8))

    for i in range(n_components):
        ax[i].plot(W[:,i],color =cols[i])

    plt.suptitle("Temporal Coefficients " + label)
    plt.show()


def plot_reconstr(reconstruct, xtest, label):
    fig, ax = plt.subplots(16, 1, figsize=(15, 15))

    for i in range(16):
        ax[i].plot(reconstruct[:, i], 'r')
        ax[i].plot(xtest[0][:, i], 'b')
        ax[i].set_ylabel(i + 1)

    plt.suptitle("Reconstruction: " + label)
    plt.show()

def match_synergies(synAE,tAE,synNNMF,tNNMF,n_components):

    new_synAE = np.copy(synAE)
    new_tAE = np.copy(tAE)
    new_synNNMF = np.copy(synNNMF)
    new_tNNMF = np.copy(tNNMF)
    all_sims = np.zeros((n_components,n_components))
    sims = np.zeros(n_components)

    for i in range(n_components):
        for j in range(n_components):
            all_sims[i,j] = np.dot(synAE[i,:], synNNMF[j,:], out=None)

    rows= np.zeros(n_components)
    cols = np.zeros(n_components)
    for i in range(n_components):
        a =  np.max(all_sims)
        sims[i] = a
        row, col = np.where(all_sims == a)
        row.astype(int)
        row = row[0]
        col.astype(int)
        col = col[0]
        for j in range(n_components):
            all_sims[row,j] = 0
            all_sims[j,col] = 0
        rows[i] = row
        cols[i] = col

    
    #reorder syn arrays
    for i in range(n_components):
        r = rows[i].astype(int)
        c = cols[i].astype(int)
        new_synAE[i,:] = synAE[r,:]
        new_tAE[:,i] = tAE[:,r]
        new_synNNMF[i,:] = synNNMF[c,:]
        new_tNNMF[:,i] = tNNMF[:,c]
        # print('similarity syn %s: '  % i + str(sims[i]))
    
    # plot_syns(new_synAE,new_tAE, "AE",n_components)
    # plot_syns(new_synNNMF,new_tNNMF,"NNMF ",n_components)

    return new_synAE,new_tAE,new_synNNMF,new_tNNMF, sims
    
def tcoef_correlations(new_tAE, new_tNNMF, n_components):
    cors = np.zeros(n_components)
    for i in range(n_components):
        # cor = np.correlate(new_tAE[:,i],new_tNNMF[:,i])
        # cors[i] = cor
        cor = np.corrcoef(new_tAE[:,i],new_tNNMF[:,i])
        cors[i] = cor[0,1]
        # print('corr tcoef %s: ' % i + str(cor))
    return cors