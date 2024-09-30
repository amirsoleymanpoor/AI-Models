#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# #DataLoading
data = pd.read_csv('file:///C:/Users/test/Desktop/DivarHousePrice.csv')

# #DataPreprocessing
data.fillna(0, inplace=True)  # Filling missing values
data = pd.get_dummies(data)  # Converting categorical features to numerical

# #FeatureSelection
# Assume 'Price' is our target variable and the rest are input features
X = data.drop('Price', axis=1)
y = data['Price']

# #DataSplitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #DataNormalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# #ANNModel
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer

# #ModelCompilation
model.compile(optimizer='adam', loss='mean_squared_error')

# #ModelTraining
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# #ModelEvaluation
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# #Predictions
predictions = model.predict(X_test)
print(predictions)


# In[ ]:




