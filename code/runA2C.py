#!/usr/bin/env python
#
# python -u runACER.py jobNumber [hyperparameters]
# The hyperparameters are listed in order below
# 'numGen', 'actorLR', 'criticLR', \
# 'lstmLayers', 'denseLayers', 'lstmUnits', 'denseUnits', \
#
#
# python -u runACER.py [TODO FILL IN]

print("starting script...")

import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import rlPulse as rlp
import spinSimulation as ss
import pandas as pd
import numpy as np
from datetime import datetime

np.seterr(all='raise')

print("imported libraries...")

print("Num GPUs Available: ", \
    len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", \
    len(tf.config.experimental.list_physical_devices('CPU')))

# define prefix for output files

prefix = f'{int(sys.argv[1]):03}-' + datetime.now().strftime("%Y%m%d-%H%M%S")

# make a new data directory if it doesn't exist
os.mkdir("../data/" + prefix)
os.mkdir("../data/" + prefix + "/weights")
output = open("../data/" + prefix + "/_output.txt", "a")
output.write(f"Output file for run {sys.argv[1]}\n\n")
print(f"created data directory, data files stored in {prefix}")

#### initialize system parameters ####

N = 4
dim = 2**N
coupling = 2*np.pi * 5e3    # coupling strength, in rad/s
delta = 2*np.pi * 500       # chemical shift strength (for identical spins)
# also in rad/s

(x,y,z) = (ss.x, ss.y, ss.z)
(X,Y,Z) = ss.getTotalSpin(N, dim)

Hdip, Hint = ss.getAllH(N, dim, coupling, delta)
HWHH0 = ss.getHWHH0(X,Y,Z,delta)

print("initialized system parameters")

# initialize RL algorithm hyperparameters

aDim = 5

numGen = 100 # how many generations to run
bufferSize = int(5e5) # size of the replay buffer
batchSize = 256 # size of batch for training, multiple of 32
gamma = .99 # future reward discount rate

actorLR = 1e-4
criticLR = 1e-3
denseLayers = 4
lstmUnits = 1024
denseUnits = 128
normalizationType = 'layer'

# save hyperparameters to output file

# hyperparameters = ['numGen', 'actorLR', 'criticLR', \
#     'lstmLayers', 'denseLayers', 'lstmUnits', 'denseUnits', \
#     'normalizationType', \
#     ]
#
# hyperparamList = "\n".join([i+": "+j for i,j in \
#     zip(hyperparameters, sys.argv[2:])])
# print(hyperparamList)
# output.write(hyperparamList)
# output.write("\n" + "="*20 + "\n"*4)
# output.flush()

# define algorithm objects

env = rlp.Environment(N, dim, coupling, delta, sDim, HWHH0, X, Y,\
    type='discrete')

#### define actor and critic networks ####

# policy function
input = layers.Input(shape=(None, aDim,), name="policyInput")
x = layers.LSTM(lstmUnits)(x)
x = layers.LayerNormalization()(x)
for l in range(denseLayers):
    y = layers.Dense(denseUnits, activation="relu")(x)
    x = layers.add([x,y])
output = layers.Dense(aDim, activation="softmax")(x)
policy = keras.Model(inputs=[stateInput], outputs=[output], name="policy")

# value function
input = layers.Input(shape=(None, aDim,), name="valueInput")
x = layers.LSTM(lstmUnits)(x)
x = layers.LayerNormalization()(x)
for l in range(denseLayers):
    y = layers.Dense(denseUnits, activation="relu")(x)
    x = layers.add([x,y])
output = layers.Dense(1)(x)
value = keras.Model(inputs=[input], outputs=[output], name="value")

# define training updates for policy and value functions

@tf.function
def policy_gradients():
    '''Calculate policy gradients using advantage function
    '''
    pass

@tf.function
def train_policy():
    '''Calculate the gradients and apply
    
    Arguments:
        s: States to train on. Size is batchSize * timesteps * aDim.
        a: Actions to train on. Size is batchSize * aDim.
        R: Total return. Size is batchSize * 1.
    '''
    with tf.GradientTape() as g:
        advantages = R - value(s)
        # TODO continue... 
    pass

# TODO and define train_step with tf.function, specify input size

# TODO fix below
# # write actor, critic summaries to output
#
# actor.model.summary(print_fn=lambda x: output.write(x + "\n"))
# output.write("\n"*2)
# critic.model.summary(print_fn=lambda x: output.write(x + "\n"))
# output.write("\n"*4)
# output.flush()

# TODO keep going here...
