from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class KerasDNN(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, optimizer='adam', dropout_rate=0.5, neurons=64, epochs=50, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.neurons = neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.neurons, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.neurons, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def fit(self, X, y, **kwargs):
        self.model = self.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_split=0.1, callbacks=[early_stopping])
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

def tune_dnn_model(X_train, y_train):
    input_dim = X_train.shape[1]
    model = KerasDNN(input_dim=input_dim, verbose=0)

    param_dist = {
        'batch_size': [20, 40],
        'epochs': [50, 100],
        'optimizer': ['adam'],
        'dropout_rate': [0.5],
        'neurons': [64]
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model