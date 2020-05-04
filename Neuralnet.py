import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
#from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

data = pd.read_csv("ABDataFinal.csv")

def normalize_data(data):
  optimal_move_columns_labels = ["move_0","move_1","move_2","move_3","move_4","move_5"]
  X = data.drop(labels=optimal_move_columns_labels, axis='columns')
  y = data[optimal_move_columns_labels]

  #normalize the input data to be between 0 and 1
  X = X.div(48)

  #reshape dataframes into np arrays to work as input for the keras model
  X = np.asarray(X)
  y = np.asarray(y)
  return X,y

 #preprocess data and load split into training and testing/input and output dataframes
#shuffle data
data = data.sample(frac=1).reset_index(drop=True)

#determine how many data points to put into training, validation, and testing sets
num_data_points = data.shape[0]
percent_train = 0.6
percent_val = 0.2
percent_test = 0.2
num_train = int(percent_train * num_data_points)
num_val = int(percent_val * num_data_points)
num_test = int(percent_test * num_data_points)

#create training, validation, and testing sets
train_data = data.head(n=num_train)
val_data = (data.head(n=num_train + num_val)).tail(n=num_val)
test_data = data.tail(n=num_test)

#normalize data
train_X, train_y = normalize_data(train_data)
val_X, val_y = normalize_data(val_data)
test_X, test_y = normalize_data(test_data)

print("There are " + str(num_data_points) + " total points in the data set")
print("There are " + str(train_X.shape[0]) + " data points in the training set")

epochs=150
batch_size = 100
model = Sequential([
                    Dense(14),
                    Dense(200, activation="relu"),
                    Dense(40, activation="relu"),
                    Dense(6, activation="softmax")
                    ])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'],
              )
history = model.fit(train_X, train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_X, val_y))

model.save('my_model')

print("Final Training Accuracy: " + str(history.history["accuracy"][-1]))
print("Final Validation Accuracy: " + str(history.history["val_accuracy"][-1]))
plt.plot(list(range(0,epochs)), history.history["accuracy"], 'b-', label='Training')
plt.plot(list(range(0,epochs)), history.history["val_accuracy"], 'g-', label='Validation')
plt.title("Accuracy over Time")
plt.legend()
plt.show()

print("Final Training Crossentropy: " + str(history.history["loss"][-1]))
print("Final Validation Crossentropy: " + str(history.history["val_loss"][-1]))
plt.plot(list(range(0,epochs)), history.history["loss"], 'b-', label='Training')
plt.plot(list(range(0,epochs)), history.history["val_loss"], 'g-', label='Validation')
plt.title("Categorical Crossentropy over Time")
plt.legend()
plt.show()


score = model.evaluate(test_X, test_y, verbose=0)
print("Testing Loss: ", score[0])
print("Testing Accuracy: ", score[1])
model.summary()
