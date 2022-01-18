import pandas as pd
import numpy as np
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import sound source A, store in form of Array and remove the first 6 column
dataframe1 = pd.read_excel("data/A.xlsx", header= None)
data1 = np.array(dataframe1)
echodata1 = np.delete(data1, [0,1,2,3,4,5], axis= 1)

# import sound source G, store in form of Array and remove the first 6 column
dataframe2 = pd.read_excel("data/G.xlsx", header= None)
data2 = np.array(dataframe2)
echodata2 = np.delete(data2, [0,1,2,3,4,5], axis= 1)

# import the TESTFILE, store in form of Array and remove the first 6 column
pred_dataframe = pd.read_excel("data/Testfile_1.xlsx", header= None)
pred_data = np.array(pred_dataframe)
x_pred = np.delete(pred_data, [0,1,2,3,4,5], axis= 1)
x_pred = np.array(x_pred, dtype= np.float)

# 2 classes for 2 signals
n_classes = 2

# Stacking both A and G togather
X = np.vstack((echodata1,echodata2))

# Labeling A as 0 and G as 1
Y = np.hstack((np.zeros(echodata1.shape[0]),np.ones(echodata2.shape[0])))

# Splitting Data and labels to test and train
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Reshaping test and train
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

# Conv1D Model for discrimination for sound sources 
model = models.Sequential(name="model_conv1D")
model.add(layers.Input(shape= x_train.shape[1:]))
model.add(layers.Conv1D(64, 7, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size= 6))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size= 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
optimizer=optimizers.RMSprop(0.001), metrics=['accuracy'])

model.summary()

# Train the model
print('Training the model:')
model.fit(x_train, y_train, epochs= 10, validation_split= 0.2, verbose= 1)

# Test the model after training
print('Testing the model:')
model.evaluate(x_test, y_test, verbose= 1)

# Confusion Matrix
y_actual = y_test
y_est = model.predict(x_test)
y_est = np.argmax(y_est, axis= 1)
cm = confusion_matrix(y_actual, y_est).ravel()
tn, fp, fn, tp  = cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm.reshape(2,2))
disp.plot()

tpr = tp/(tp + fn)
tnr = tn/(tn + fp)
fdr = fp/(fp + tp)
npv = tn/(tn + fn)

# Predict the test data
print('Prediction using trained model:')
y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis= 1)
D = np.where(y_pred > 0.5, 'G', 'A')
print (D)
