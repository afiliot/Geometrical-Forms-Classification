# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
import keras
from keras.callbacks import Callback
from IPython.display import clear_output

# On some implementations of matplotlib, you may need to change this value
IMAGE_SIZE = 72
FIGSIZE = IMAGE_SIZE/72

def generate_a_drawing(U, V, noise=0.0):
    fig = plt.figure(figsize=(FIGSIZE,FIGSIZE))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,FIGSIZE)
    ax.set_ylim(0,FIGSIZE)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False):
    figsize = FIGSIZE  
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(U, V, noise)


def generate_a_disk(noise=0.0, free_location=False):
    figsize = FIGSIZE
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = FIGSIZE
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(U, V, noise)
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]

def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros(nb_samples)
    for i in tqdm(range(nb_samples), desc='Creating data for classification'):
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1: 
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

def generate_test_set_classification(nb_samples, noise=20.0, free_location=False):
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(nb_samples, noise, free_location)
    Y_test = keras.utils.to_categorical(Y_test, 3) 
    return [X_test, Y_test]

def generate_dataset_regression(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples, 6])
    for i in tqdm(range(nb_samples), desc='Creating data for regression'):
        [X[i], Y[i]] = generate_a_triangle(noise, free_location)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

import matplotlib.patches as patches

def generate_test_set_regression(nb_samples=300, noise=20.0, free_location=False):
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(nb_samples, noise, free_location)
    return [X_test, Y_test]

def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((IMAGE_SIZE,IMAGE_SIZE))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)
    
def plot_metrics(epochs, metrics, metric_name):
    if metric_name == 'accuracy':
        m = 'acc'
    elif metric_name == 'PSNR':
        m = metric_name
    elif metric_name == 'mean_absolute_error':
        m = metric_name
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs, metrics[m], label='Train', marker='.', linestyle='-')
    plt.plot(epochs, metrics['val_'+m], label='Val', marker='.', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(metric_name+' with epochs')
    plt.legend()
    plt.subplot(1,2,2)
    plt.semilogy(epochs, metrics['loss'], label='Train', marker='.', linestyle='-')
    plt.semilogy(epochs, metrics['val_loss'], label='Val', marker='.', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss with epochs')
    plt.legend()
    plt.show()
    
def generate_a_drawing_denoising(U, V, noise=30.0):
    fig = plt.figure(figsize=(FIGSIZE,FIGSIZE))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,FIGSIZE)
    ax.set_ylim(0,FIGSIZE)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata_noised = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata_noised, imdata

def generate_a_rectangle_denoising(noise=30.0, free_location=True):
    figsize = FIGSIZE
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing_denoising(U, V, noise)


def generate_a_disk_denoising(noise=30.0, free_location=True):
    figsize = FIGSIZE
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing_denoising(U, V, noise)

def generate_a_triangle_denoising(noise=30.0, free_location=True):
    figsize = FIGSIZE
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    return generate_a_drawing_denoising(U, V, noise)

def generate_dataset_denoising(nb_samples, noise=30.0, free_location=True):
    # Getting im_size:
    im_size = generate_a_rectangle_denoising()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples,im_size])
    for i in tqdm(range(nb_samples), desc='Creating data for classification'):
        category = np.random.randint(3)
        if category == 0:
            X[i], Y[i] = generate_a_rectangle_denoising(noise, free_location)
        elif category == 1: 
            X[i], Y[i] = generate_a_disk_denoising(noise, free_location)
        else:
            X[i], Y[i] = generate_a_triangle_denoising(noise, free_location)
    X = (X + noise) / (255 + 2 * noise)
    Y = Y / 255
    return [X, Y]