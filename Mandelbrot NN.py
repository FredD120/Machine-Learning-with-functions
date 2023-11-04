#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Mandelbrot GIF


# In[114]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(i):
    mult = zoom**i 
    new_xmin=xmin*mult -1.405
    new_xmax=xmax*mult -1.405
    new_ymin=ymin*mult
    new_ymax=ymax*mult
    
    mandelbrot_img = mandelbrot_set(new_xmin, new_xmax, new_ymin, new_ymax, width, height, max_iter)
    mandel = plt.imshow(mandelbrot_img.T, extent=(xmin, xmax, ymin, ymax), cmap='hot', origin='lower')

@jit
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return maxiter

def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    img = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            img[i,j] = mandelbrot(x[i],y[j],maxiter)
    return img

# Define the coordinates and rendering parameters
xmin, xmax, ymin, ymax = -0.6, 1, -0.5, 0.5
width, height = 1000, 1000
max_iter = 250
frame = 60
zoom=0.8

mandelbrot_img= mandelbrot_set(xmin-1.4, xmax-1.4, ymin, ymax, width, height, max_iter)

# Display the Mandelbrot set
f0=plt.figure(figsize=(10,10))
mandel = plt.imshow(mandelbrot_img.T, extent=(xmin, xmax, ymin, ymax), cmap='hot', origin='lower')
ani = animation.FuncAnimation(fig=f0, func=animate, frames=frame, interval=100)
plt.axis('off')
plt.show()

savefile = r"Mandelbrot.gif"
pillowwriter = animation.PillowWriter(fps=2)
ani.save(savefile, writer=pillowwriter)


# In[ ]:


#Image generation and Training


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Add
from tensorflow.keras.callbacks import LambdaCallback
from numba import jit

def generating_func(x,y):
    return np.sin(5*(x**2+y**2))

def generate_image(xmin, xmax, ymin, ymax, width, height):
    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    xv, yv = np.meshgrid(x, y)
    img = generating_func(xv, yv)
    
    img = img + abs(np.amin(img))
    return img / np.amax(img)

def generate_inputs(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    xs = np.repeat(x, height)
    ys = np.tile(y, width)
    #input_pairs = np.column_stack((xs, ys))
    
    return [xs,ys]

# Define the coordinates of training data
xmin, xmax, ymin, ymax = -1,1,-1,1
width, height = 50, 50


# Generate the answers to training inputs
training_img = generate_image(xmin, xmax, ymin, ymax, width, height)
training_inputs = generate_inputs(xmin, xmax, ymin, ymax, width, height)


#NN setup
input1 = Input(shape=(1,), name="x_coordinate")
input2 = Input(shape=(1,), name="y_coordinate")

branch1 = Dense(32, activation='relu')(input1)
branch2 = Dense(32, activation='relu')(input2)

concatenated = Concatenate()([branch1, branch2])

comp_layer1 = Dense(64, activation='relu')(concatenated)
comp_layer2 = Dense(64, activation='relu')(comp_layer1)

output = Dense(1, activation='sigmoid', use_bias=True)(comp_layer2)
model = Model(inputs=[input1, input2], outputs=output)

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

num_epochs = 5

#Create array of predictions
#predictions = np.zeros((len(training_img.flatten()),num_epochs))

def on_epoch_end(epoch,logs):
    predictions[:,epoch] = model.predict(training_inputs)[:,0]

        
predict_callback = LambdaCallback(on_epoch_end=on_epoch_end)
#model.fit(training_inputs, training_img.flatten(), epochs=num_epochs, callbacks=[predict_callback])
model.fit(training_inputs, training_img.flatten(),epochs=num_epochs,batch_size=1)

model.save("radial_image_model")

f0=plt.figure(figsize=(10,10))
plt.imshow(training_img, extent=(xmin, xmax, ymin, ymax), cmap='plasma', origin='lower')
plt.axis('off')
plt.show()


# In[ ]:


#NN testing on image data


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Add
from tensorflow.keras.callbacks import LambdaCallback
from numba import jit

def generating_func(x,y):
    return np.sin(x**2+y**2)

def generate_image(xmin, xmax, ymin, ymax, width, height):
    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    xv, yv = np.meshgrid(x, y)
    img = generating_func(xv, yv)
    
    img = img + abs(np.amin(img))
    return img / np.amax(img)

def generate_inputs(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    xs = np.repeat(x, height)
    ys = np.tile(y, width)
    
    return [xs,ys]

# Define the coordinates of training data
xmin, xmax, ymin, ymax = -1,1,-1,1
width, height = 50, 50

model = keras.models.load_model("radial_image_model")

test_inputs = generate_inputs(1.2*xmin, 1.2*xmax, 1.2*ymin, 1.2*ymax, int(1.2*width), int(1.2*height))

test_answers = generate_image(1.2*xmin, 1.2*xmax, 1.2*ymin, 1.2*ymax, int(1.2*width), int(1.2*height))

predictions = model.predict(test_inputs)
predictions = predictions.reshape(( int(1.2*width),int(1.2*height)))

# Display the data
f0=plt.figure(figsize=(10,10))

plt.imshow(predictions, extent=(xmin, xmax, ymin, ymax), cmap='plasma', origin='lower')
plt.axis('off')
plt.show()


# In[ ]:


#Mandelbrot NN Training


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Add
from tensorflow.keras.callbacks import LambdaCallback
from numba import jit

@jit
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return maxiter

def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    img = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            img[i,j] = mandelbrot(x[i],y[j],maxiter)
    return img.T / maxiter


def generate_inputs(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    xs = np.repeat(x, height)
    ys = np.tile(y, width)
    #input_pairs = np.column_stack((xs, ys))
    
    return [xs,ys]

# Define the coordinates of training data
xmin, xmax, ymin, ymax = -1.5, 0.5, -0.9, 0.9
width, height = 1000, 1000
max_iter = 250


# Generate the answers to training inputs
training_img = mandelbrot_set(xmin, xmax, ymin, ymax, width, height,max_iter)
training_inputs = generate_inputs(xmin, xmax, ymin, ymax, width, height)


#NN setup
input1 = Input(shape=(1,), name="x_coordinate")
input2 = Input(shape=(1,), name="y_coordinate")

branch1 = Dense(128, activation='relu')(input1)
branch2 = Dense(128, activation='relu')(input2)

concatenated = Concatenate()([branch1, branch2])

comp_layer1 = Dense(256, activation='relu')(concatenated)
comp_layer2 = Dense(256, activation='relu')(comp_layer1)
comp_layer3 = Dense(256, activation='relu')(comp_layer2)
comp_layer4 = Dense(256, activation='relu')(comp_layer3)
comp_layer5 = Dense(256, activation='relu')(comp_layer4)

output = Dense(1, activation='sigmoid', use_bias=True)(comp_layer5)
model = Model(inputs=[input1, input2], outputs=output)

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

num_epochs = 10
        
model.fit(training_inputs, training_img.flatten(),epochs=num_epochs)

model.save("mandelbrot_model_V2")

f0=plt.figure(figsize=(10,10))
plt.imshow(training_img, extent=(xmin, xmax, ymin, ymax), cmap='hot', origin='lower')
plt.axis('off')
plt.show()


# In[8]:


#Mandelbrot NN Testing


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Add
from tensorflow.keras.callbacks import LambdaCallback
from numba import jit

@jit
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return maxiter

def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    img = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            img[i,j] = mandelbrot(x[i],y[j],maxiter)
    return img.T


def generate_inputs(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    xs = np.repeat(x, height)
    ys = np.tile(y, width)
    #input_pairs = np.column_stack((xs, ys))
    
    return [xs,ys]

# Define the coordinates of training data
xmin, xmax, ymin, ymax = -1.5, 0.5, -0.9, 0.9
width, height = 1000, 1000
max_iter = 250

model = keras.models.load_model("mandelbrot_model")

test_inputs = generate_inputs(1*xmin, 1*xmax, 1*ymin, 1*ymax, int(1*width), int(1*height))

test_answers = mandelbrot_set(1*xmin, 1*xmax, 1*ymin, 1*ymax, int(1*width), int(1*height),max_iter)

predictions = model.predict(test_inputs)
predictions = predictions.reshape(( int(1*width),int(1*height)))

# Display the data
f0=plt.figure(figsize=(10,10))

plt.imshow(predictions, extent=(xmin, xmax, ymin, ymax), cmap='hot', origin='lower')
plt.axis('off')
plt.show()


# In[ ]:


#Testing the NN on a zoomed in section of the mandelbrot set


# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Add
from tensorflow.keras.callbacks import LambdaCallback
from numba import jit

@jit
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return maxiter

def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    img = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            img[i,j] = mandelbrot(x[i],y[j],maxiter)
    return img.T


def generate_inputs(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    xs = np.repeat(x, height)
    ys = np.tile(y, width)
    #input_pairs = np.column_stack((xs, ys))
    
    return [xs,ys]

# Define the coordinates of training data
xmin, xmax, ymin, ymax = -1.5, -0.8, -0.35, 0.35
width, height = 1000, 1000
max_iter = 250

model = keras.models.load_model("mandelbrot_model")

test_inputs = generate_inputs(1*xmin, 1*xmax, 1*ymin, 1*ymax, int(1*width), int(1*height))

predictions = model.predict(test_inputs)
predictions = predictions.reshape(( int(1*width),int(1*height)))

test_answers = mandelbrot_set(1*xmin, 1*xmax, 1*ymin, 1*ymax, int(1*width), int(1*height),max_iter)


f0=plt.figure(figsize=(10,10))

plt.imshow(predictions, extent=(xmin, xmax, ymin, ymax), cmap='hot', origin='lower')
plt.axis('off')
plt.show()

f1=plt.figure(figsize=(10,10))

plt.imshow(test_answers, extent=(xmin, xmax, ymin, ymax), cmap='hot', origin='lower')
plt.axis('off')
plt.show()


# In[ ]:




