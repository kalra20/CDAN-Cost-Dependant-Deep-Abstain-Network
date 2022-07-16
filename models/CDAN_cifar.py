from __future__ import print_function

import keras
import numpy as np
import os
import pickle
from keras import backend as K
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.constraints import max_norm
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
from CDAN_utils import *
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10 as cifar10
#tf.enable_eager_execution()


class CIFARVgg:
    def __init__(self, train=True, filename="weightsvggnt.h5", coverage=0.8, alpha=0.5, baseline=False,d=0.9,mu=1.0,epochs=250,lr=1e-2):
        
        self.mu = mu
        self.alpha = alpha
        self.lr = lr
        self.num_classes = 10
        self.weight_decay = 5e-4
        self.weight_decay_fc = 5e-4
        #self._load_ddecay_fc = 1e-7
        self.weight_decay_rc = 5e-4
        self._load_data()
        self.d = d
        self.rho = tf.Variable(1.0,trainable=True)
        # self.g_shape = self.g.shape[1:]
        self.epochs = epochs
        self.x_shape = self.x_train.shape[1:]
        print("mu value:",self.mu)
        self.filename = "weightsvgg_100_64_r_05.h5"

        self.model = self.build_model()
        
        if train:
            print('loaded_weights here')
            self.model = self.train(self.model)
        else:
            print('loaded_weights nhere')
            self.model.load_weights("checkpoints3/{}".format(self.filename))

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        # acti_2 = Activation(custom_activation)
        acti = 'relu'
        weight_decay = self.weight_decay
        weight_decay_fc = self.weight_decay_fc
        weight_decay_rc = self.weight_decay_rc
        basic_dropout_rate = 0.3
        inputa = Input(shape=self.x_shape)
        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block1_conv1',trainable=True)(inputa)
        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate)(curr)

        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block1_conv2',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block1_pool',trainable=True)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block2_conv1',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block2_conv2',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block2_pool',trainable=True)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block3_conv1',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block3_conv2',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block3_conv3',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block3_pool',trainable=True)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block4_conv1',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block4_conv2',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block4_conv3',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block4_pool',trainable=True)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block5_conv1',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block5_conv2',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block5_conv3',trainable=True)(curr)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block5_pool',trainable=True)(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr_1 = Flatten()(curr)
        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay_fc),activation=acti,trainable=True)(curr_1)

        #curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)

        curr_a = Dropout(basic_dropout_rate + 0.2)(curr)

        curr1 = Dense(self.num_classes,kernel_regularizer=regularizers.l2(weight_decay_rc))(curr_a)

        
        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay_rc),activation=acti,trainable=True)(curr_1)
        ##curr = Activation('relu')(curr)
        curr2 = BatchNormalization()(curr2)
        curr2 = Dropout(basic_dropout_rate + 0.2)(curr2)

        curr2 = Dense(self.num_classes,kernel_regularizer=regularizers.l2(weight_decay_rc),activation = 'sigmoid')(curr2)
        
        # auxiliary head (h)
        CDAN_output = concatenate([curr1, curr2],name='dr_head')

        auxiliary_output = Dense(self.num_classes, activation='softmax', name="classification_head")(curr_a)

        #auxiliary_output = Dense(1, name="classification_head")(curr)
        self.model = Model(inputs=[inputa], outputs=[CDAN_output,auxiliary_output])

        return self.model

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def predict(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
            # g = self.g2
        return self.model.predict([x,g], batch_size)

    def predict_embedding(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model_embeding.predict(x, batch_size)

    def mc_dropout(self, batch_size=1000, dropout=0.5, iter=100):
        K.set_value(self.mc_dropout_rate, dropout)
        repititions = []
        for i in range(iter):
            _, pred = self.model.predict(self.x_test, batch_size)
            repititions.append(pred)
        K.set_value(self.mc_dropout_rate, 0)

        repititions = np.array(repititions)
        mc = np.var(repititions, 0)
        mc = np.mean(mc, -1)
        return -mc


    def _load_data(self):

        # The data, shuffled and split between train and test sets:
        # (x_train, y_train), (x_test, y_test_label) = load_cats_vs_dogs()
        # mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # print(y_train[0:100])
        # (x_train,y_train) = self.binary_mnist(x_train,y_train)
        # (x_test,y_test_label) = self.binary_mnist(x_test,y_test)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        self.x_train, self.x_test = self.normalize(x_train, x_test)
        
        
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)
        print(self.y_test.shape)


    def train(self, model):
        #c = self.lamda
        #lamda = 32


        def double_ramp_loss(y_true,y_pred):
            y_true_bool = tf.cast(y_true,dtype=tf.bool)
            print("y_true bool",y_true_bool)
            y_true_c = tf.cast(tf.logical_not(y_true_bool),dtype=tf.float32)
            f_y = tf.keras.backend.max(tf.multiply(y_true,y_pred[:,0:self.num_classes]),axis=-1)
            print('f_y',f_y)
            f_r = tf.keras.backend.max(tf.multiply(y_true_c,y_pred[:,0:self.num_classes]),axis=-1)
            rho_y = tf.keras.backend.max(tf.multiply(y_true,y_pred[:,self.num_classes:]),axis=-1)
            loss = self.d*self.mu**-1*(tf.keras.activations.relu(self.mu-f_y+f_r+rho_y)-tf.keras.activations.relu(-self.mu**2-f_y+f_r+rho_y))+(1-self.d)*self.mu**-1*(tf.keras.activations.relu(self.mu-f_y+f_r-rho_y)-tf.keras.activations.relu(-self.mu**2-f_y+f_r-rho_y))
            #loss = 2*self.d*tf.math.sigmoid(-1*(f_y-f_r-rho_y)) + 2*(1-self.d)*tf.math.sigmoid(-1*(f_y-f_r+rho_y))
            print('loss',loss)
            loss = K.mean(loss,axis=0)
            return loss
        

        def double_ramp_accuracy(y_true,y_pred):
            y_pred = K.cast(y_pred,tf.float32)

            y_true_bool = tf.cast(y_true,dtype=tf.bool)
            y_true_c = tf.cast(tf.logical_not(y_true_bool),dtype=tf.float32)
            f_y = tf.keras.backend.max(tf.multiply(y_true,y_pred[:,0:self.num_classes]),axis=-1)
            f_r = tf.keras.backend.max(tf.multiply(y_true_c,y_pred[:,0:self.num_classes]),axis=-1)
            rho_y = tf.keras.backend.max(tf.multiply(y_true,y_pred[:,self.num_classes:]),axis=-1)
            r5 = tf.reduce_sum(tf.cast(tf.math.greater(f_y-rho_y,f_r),dtype=tf.float32))
            r6 = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.greater_equal(f_y+rho_y,f_r),tf.math.less(f_y-rho_y,f_r)),tf.float32))
            tn0 = tf.reduce_sum(tf.cast(tf.math.logical_or(tf.math.greater(f_y-rho_y,f_r),tf.math.less_equal(f_y-rho_y,f_r)),tf.float32))
            print('y_pred_mul',y_pred[:,self.num_classes:])
            

            acc = r5*(tn0-r6)**-1

            #acc = r5/tn1
            return tf.cast(acc,tf.float32)
            # return tf.cast(100,tf.float32)
        def rho_acc(y_true,y_pred):
            y_pred = K.cast(y_pred,tf.float32)

            y_true_bool = tf.cast(y_true,dtype=tf.bool)
            y_true_c = tf.cast(tf.logical_not(y_true_bool),dtype=tf.float32)
            f_y = tf.keras.backend.max(tf.multiply(y_true,y_pred[:,0:self.num_classes]),axis=-1)
            f_r = tf.keras.backend.max(tf.multiply(y_true_c,y_pred[:,0:self.num_classes]),axis=-1)
            rho_y = tf.keras.backend.max(tf.multiply(y_true,y_pred[:,self.num_classes:]),axis=-1)
            r5 = tf.reduce_sum(tf.cast(tf.math.greater(f_y-rho_y,f_r),dtype=tf.float32))
            r6 = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.greater_equal(f_y+rho_y,f_r),tf.math.less(f_y-rho_y,f_r)),tf.float32))
            tn0 = tf.reduce_sum(tf.cast(tf.math.logical_or(tf.math.greater(f_y-rho_y,f_r),tf.math.less_equal(f_y-rho_y,f_r)),tf.float32))
            acc = r6/tn0

            #acc = tn0*(r5/(tn0-r6))*100**-1
            return tf.cast(acc,tf.float32)

        

        # training parameters
        batch_size = 128
        maxepoches = self.epochs
        learning_rate = self.lr

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_dr_head_loss',factor=0.5,patience=20,min_lr=0.00001,min_delta=0.0001,verbose=1)
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=15,min_lr=0.00001,min_delta=0.001,verbose=1)
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=15,min_lr=0.00001,min_delta=0.001,verbose=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=60,min_delta=0.0001)
        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.x_train)
        ep = 1e-07
        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        #sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        #adagrad = optimizers.Adagrad(lr=learning_rate,epsilon=ep)
        adam = optimizers.Adam(lr=learning_rate,epsilon=ep)
        model.compile(loss=[double_ramp_loss,'categorical_crossentropy'],loss_weights=[self.alpha,1-self.alpha],
                      optimizer=sgd, metrics={'dr_head':[ double_ramp_accuracy,rho_acc],'classification_head':['accuracy']})
        print(self.x_train.shape[0])
        historytemp = model.fit_generator(my_generator(datagen.flow,self.x_train,self.y_train,batch_size,2), use_multiprocessing = False,
                                          steps_per_epoch=self.x_train.shape[0] // batch_size,
                                          epochs=maxepoches, callbacks=[reduce_lr],validation_steps=5,
                                        initial_epoch=0,
                                        validation_data=([self.x_test], [self.y_test,self.y_test]))
        #self.filename = "weightsvgg_05.h5"
        with open("checkpoints/{}_history.pkl".format(self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model.save_weights("checkpoints/{}".format(self.filename))

        return model

