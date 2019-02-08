
# coding: utf-8

# In[4]:


import numpy as np
from keras.models import Model
from keras.layers import Bidirectional,Dense, Input, Dropout, LSTM, Activation, GRU, Conv1D,ConvLSTM2D, Flatten, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


class RNN():

    model = []
    history = []
    epochs = 100

    def LSTMModel(self, input_shape, embedding_layer):
        """
        Creates a LSTM Model using keras

        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        embedding_layer -- embedding layer using word2vec

        %65.8 acc 40 iterations
        """

        # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
        sentence_indices = Input(shape=input_shape, dtype=np.int32)

        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        embeddings =  embedding_layer(sentence_indices)

        X = Bidirectional(LSTM(128, return_sequences=False, dropout=0.8, recurrent_dropout=0.5))(embeddings)
        #X = GlobalMaxPool1D()(X)
        X = Dense(128, activation="relu")(X)
        X = Dropout(0.8)(X)
        X = Dense(3, activation="sigmoid")(X)

        # Create Model instance which converts sentence_indices into X.
        self.model = Model(sentence_indices, X)

    def LSTMConvModel(self, input_shape, embedding_layer):
        """
        Creates a LSTM Model using keras

        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        embedding_layer -- embedding layer using word2vec

        % 66 % 0.01 rate 0.5 dropout
        % 65 0.0001 rate 0.5 dropout
        % 65 0.001  rate 0.5 dropout 512 minibatch
        """

        # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
        sentence_indices = Input(shape=input_shape, dtype=np.int32)

        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        embeddings =  embedding_layer(sentence_indices)

        #0.35
        #X = SpatialDropout1D(0.35)(embeddings)
        X = LSTM(128, return_sequences=True,kernel_initializer='glorot_uniform' )  (embeddings)
        X = Dropout(0.5)(X)
        #X = Conv1D(64, kernel_size=3,,kernel_regularizer=regularizers.l2(0.01) padding='valid', kernel_initializer='glorot_uniform')(X)
        #kernel_regularizer=regularizers.l2(0.01)
        X = LSTM(128, return_sequences=False,kernel_initializer='glorot_uniform')  (X)
        X = Dropout(0.5)(X)
        #X = Dropout(0.5)(X)
        # Add a softmax activation
        X = Dense(3,activation='softmax')(X)


        # Create Model instance which converts sentence_indices into X.
        self.model = Model(sentence_indices, X)



    def CompileModel(self):

        a = 0.001
        opt = Adam(lr=a, epsilon=None, decay=0, amsgrad=False)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[categorical_accuracy])


    def FitModel(self,trainingX,trainingY,testX,testY):

        #checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_categorical_accuracy',save_best_only=True, mode='max')
        #self.history = self.model.fit(trainingX, trainingY, epochs = self.epochs,batch_size = 512, shuffle=True,validation_data=(testX, testY) ,callbacks=[checkpoint],verbose=2)
        self.history = self.model.fit(trainingX, trainingY, epochs = self.epochs,batch_size = 256, shuffle=True,validation_data=(testX, testY),verbose=1)
        self.GenerateGraphics()

    def Predict(self):
        self.model.predict()

    def ModelSumary(self):
        print(self.model.summary())

    def GenerateGraphics(self):
        # plot the training loss and accuracy
        plt.clf()
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), self.history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["categorical_accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_categorical_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()
