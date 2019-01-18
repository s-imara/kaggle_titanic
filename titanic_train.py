from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential

# Generate dummy data
import data

x_train = data.train_data_x
y_train = data.train_data_y
#x_test = np.random.random((100, 4))
#y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(128, input_dim=6, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(64,  activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(32,  activation='sigmoid'))
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))
opt = optimizers.Adam(lr=0.00045)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#callback = EarlyStopping(monitor='val_loss', verbose=1)
model.fit(x_train, y_train,
          epochs=10000,
          validation_split=30,
          batch_size=64)
          #,callbacks=[callback])
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
