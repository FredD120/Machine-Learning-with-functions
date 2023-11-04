#!/usr/bin/env python
# coding: utf-8

# In[53]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(1)

def animate(i):
    ModelPredicted.set_ydata(predictions[:,i])
    label = "Epochs "+str(i+1)
    L.get_texts()[0].set_text(label)
    return ModelPredicted

def generating_func(x,var): #Equation to be guessed by NN
    result = (x+1)*(x-20)*(x+45)*(x-39)*x*(1+var*0)
    return result/np.max(result)

lower_bound = -50
upper_bound = 50
num_points = (upper_bound-lower_bound)*1000
x_axis = np.linspace(lower_bound,upper_bound,num_points) #True x axis
correct_line = np.zeros(x_axis.shape)
correct_line[:] = generating_func(x_axis[:],correct_line[:]) #True data

interval = 5
input_error = 10 #As a percentage
sparse_x_axis = x_axis[1::interval] #Training data
input_data = np.zeros(sparse_x_axis.shape)
random_offsets = np.random.uniform(-0.5*(input_error/100),0.5*(input_error/100),sparse_x_axis.shape) 
input_data[:] = generating_func(sparse_x_axis[:],random_offsets[:]) #Actual output

test_interval = 3
test_x_axis = x_axis[1::test_interval] #Test data
test_data = np.zeros(test_x_axis.shape)
test_data[:] = generating_func(test_x_axis[:],0) #Test answers


model = keras.Sequential()
model.add(Input(shape=(1,), name = 'Input_Layer'))
model.add(Dense(500, activation='relu', name = 'Hidden_Layer1'))
model.add(Dense(500, activation='relu', name = 'Hidden_Layer2'))
model.add(Dense(500, activation='relu', name = 'Hidden_Layer3'))
model.add(Dense(500, activation='relu', name = 'Hidden_Layer4'))
model.add(Dense(500, activation='relu', name = 'Hidden_Layer5'))
model.add(Dense(500, activation='relu', name = 'Hidden_Layer6'))
model.add(Dense(1, activation='tanh', name = 'Output_Layer'))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

num_epochs = 10
predictions = np.zeros((len(test_x_axis),num_epochs))

def on_epoch_end(epoch,logs):
    predictions[:,epoch] = model.predict(test_x_axis)[:,0]
        
predict_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(sparse_x_axis, input_data, epochs=num_epochs, callbacks=[predict_callback])


f0,ax = plt.subplots(figsize = (10,6))

#SparseValues = plt.scatter(sparse_x_axis,input_data,color='blue',label = 'True Value', s = 3)
ModelPredicted = plt.plot(test_x_axis,predictions[:,0],color='green',label = 'Epoch 1',alpha=1,linewidth=3)[0]
TrueValues = plt.plot(x_axis,correct_line,color='blue',label = 'True Value',linestyle='dashed',alpha=0.5,linewidth=2)
ani = animation.FuncAnimation(fig=f0, func=animate, frames=num_epochs, interval=1000)
#ax.set(xlim=[lower_bound, upper_bound], ylim=[-1.1, 1.1], xlabel='x', ylabel='y')
writer = animation.PillowWriter(fps=1, metadata=dict(artist='Me'),bitrate=1800)                                                             
L=ax.legend(loc=9)
ani.save('NN_Learning1D_V2.gif', writer=writer)


# In[ ]:





# In[ ]:




