# Create neural net
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from utils import to_xy
x, y = 
model = Sequential()

model.add(Dense(10, input_dim=x.shape[1],
                kernel_initializer='normal',
                activation='relu'))

model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='normal',
                activation='relu'))
model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)