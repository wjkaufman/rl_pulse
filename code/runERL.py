#!/usr/bin/env python
#
# python runERL.py jobNumber [hyperparameters]
# The hyperparameters are listed in order below
# numGen, bufferSize, batchSize, popSize, polyak, gamma, syncEvery,
# actorLR, criticLR, lstmLayers, fcLayers, lstmUnits, fcUnits
#
# python runERL.py 1 2 1000 100 10 .01 .99 2 .001 .01 1 4 32 32

print("starting runRLPulse script...")

import sys
import os
import rlPulse as rlp
import spinSimulation as ss
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("imported libraries...")

# define prefix for output files

prefix = datetime.now().strftime("%Y%m%d-%H%M%S") + f'-{int(sys.argv[1]):03}'

# make a new data directory if it doesn't exist
os.mkdir("../data/" + prefix)
output = open("../data/" + prefix + "/output.txt", "a")
output.write(f"Output file for run {sys.argv[1]}\n\n")
print(f"created data directory, data files stored in {prefix}")

# initialize system parameters

N = 4
dim = 2**N
coupling = 5e3    # coupling strength
delta = 500       # chemical shift strength (for identical spins)

(x,y,z) = (ss.x, ss.y, ss.z)
(X,Y,Z) = ss.getTotalSpin(N, dim)

Hdip, Hint = ss.getAllH(N, dim, coupling, delta)
HWHH0 = ss.getHWHH0(X,Y,Z,delta)

print("initialized system parameters")

# initialize RL algorithm hyperparameters

sDim = 3 # state represented by sequences of actions...
aDim = 3 # action = [phi, rot, time]

numGen = int(sys.argv[2]) # how many generations to run
bufferSize = int(sys.argv[3]) # size of the replay buffer
batchSize = int(sys.argv[4]) # size of batch for training
popSize = int(sys.argv[5]) # size of population
polyak = float(sys.argv[6]) # polyak averaging parameter
gamma = float(sys.argv[7]) # future reward discount rate

# start updating actor/critic networks after this many episodes
# updateAfter = int(sys.argv[12])
syncEvery = int(sys.argv[8]) # how often to copy RL actor into population
numTests = 50

p = .05

actorLR = float(sys.argv[9])
criticLR = float(sys.argv[10])
lstmLayers = int(sys.argv[11])
fcLayers = int(sys.argv[12])
lstmUnits = int(sys.argv[13])
fcUnits = int(sys.argv[14])

# save hyperparameters to output file

hyperparameters = ['numGen', 'bufferSize', 'batchSize', 'popSize', 'polyak', \
    'gamma', 'syncEvery', 'actorLR', 'criticLR', 'lstmLayers', 'fcLayers', \
    'lstmUnits', 'fcUnits']

hyperparamList = "\n".join([i+": "+j for i,j in zip(hyperparameters, sys.argv[2:])])
print(hyperparamList)
output.write(hyperparamList)
output.write("\n" + "="*20 + "\n"*4)
output.flush()

# define algorithm objects

actor = rlp.Actor(sDim,aDim, actorLR)
actorTarget = rlp.Actor(sDim,aDim, actorLR)
critic = rlp.Critic(sDim, aDim, gamma, criticLR)
criticTarget = rlp.Critic(sDim, aDim, gamma, criticLR)

actor.createNetwork(lstmLayers, fcLayers, lstmUnits, fcUnits)
actorTarget.createNetwork(lstmLayers, fcLayers, lstmUnits, fcUnits)
critic.createNetwork(lstmLayers, fcLayers, lstmUnits, fcUnits)
criticTarget.createNetwork(lstmLayers, fcLayers, lstmUnits, fcUnits)

actorTarget.setParams(actor.getParams())
criticTarget.setParams(critic.getParams())

env = rlp.Environment(N, dim, coupling, delta, sDim, HWHH0, X, Y)

pop = rlp.Population(popSize)
pop.startPopulation(sDim, aDim, actorLR, lstmLayers, fcLayers, \
    lstmUnits, fcUnits)

noise = rlp.NoiseProcess(p)

replayBuffer = rlp.ReplayBuffer(bufferSize)

# write actor, critic summaries to output

actor.model.summary(print_fn=lambda x: output.write(x + "\n"))
output.write("\n"*2)
critic.model.summary(print_fn=lambda x: output.write(x + "\n"))
output.write("\n"*4)
output.flush()

# ERL algorithm

# record test results: generation, final pulse sequence, rewards
testFile = open("../data/"+prefix+"/testResults.txt", 'a')
testFile.write("Test results\n\n")

print(f"starting ERL algorithm ({datetime.now()})")
output.write(f"started ERL algorithm: {datetime.now()}\n")

for i in range(numGen):
    print(f"On generation {i} ({datetime.now()})")
    
    # evaluate and iterate the population
    pop.evaluate(env, replayBuffer, None, numEval=5)
    print("evaluated population")
    pop.iterate() # TODO a lot of hyperparameters in here to tune...
    print("iterated population")
    
    # evaluate the actor
    actor.evaluate(env, replayBuffer, noise)
    print("evaluated the actor")
    
    # update networks
    batch = replayBuffer.getSampleBatch(batchSize)
    # train critic
    critic.trainStep(batch, actorTarget, criticTarget)
    # train actor
    actor.trainStep(batch, critic)
    # update target networks
    criticTarget.copyParams(critic, polyak)
    actorTarget.copyParams(actor, polyak)
    
    print("trained actor/critic")
    
    if i % int(np.ceil(numGen / numTests)) == 0:
        print(f"Recording test results (generation {i})")
        s, rMat = actor.test(env)
        # record results from the test
        print(f"Max reward from test: {np.max(rMat):0.02f}")
        testFile.write(f"Test result from generation {i}\n\nChosen pulse sequence:\n")
        testFile.write(rlp.formatAction(s) + "\n")
        testFile.write("Rewards from the pulse sequence:\n")
        for testR in rMat:
            testFile.write(f"{testR:.02f}, ")
        testFile.write("\n"*3)
        testFile.flush()
    
    print(f'buffer size is {replayBuffer.size}')
            
print(f"finished ERL algorithm ({datetime.now()})")
output.write(f"finished ERL algorithm: {datetime.now()}\n\n")
output.write("="*50 + "\n")

testFile.flush()
testFile.close()

# results (save everything to files)

# TODO put in other results...

# calculate other benchmarks of run

rBuffer = np.array([_[2] for _  in replayBuffer.buffer])
indSorted = rBuffer.argsort()
for i in range(1,5):
    output.write(f"Highest rewards in buffer (#{i})\n")
    output.write(f"Index in buffer: {indSorted[-i]}\n")
    sequence = replayBuffer.buffer[indSorted[-i]][3] # sequence of actions
    output.write(rlp.formatAction(sequence) + "\n")
    # calculate mean fidelity from ensemble of dipolar couplings
    fidelities = np.zeros((10,))
    t = np.sum(rlp.getTimeFromAction(sequence))
    for i in range(10):
        Hdip, Hint = ss.getAllH(N, dim, coupling, delta)
        Uexp = rlp.getPropagatorFromAction(N, dim, sequence, Hint, X, Y)
        Utarget = ss.getPropagator(HWHH0, t)
        fidelities[i] = ss.fidelity(Utarget, Uexp)
    fMean = np.mean(fidelities)
    output.write(f"Mean fidelity: {fMean}\n")
    r = -1*np.log10(1+1e-12-fMean**(20e-6/t))
    output.write(f"Reward: {r:.03}\n")

output.write("="*20 + "\n\n")

output.flush()
output.close()

print("finished running script!")
