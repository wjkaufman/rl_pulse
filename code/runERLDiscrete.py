#!/usr/bin/env python
#
# python -u runERL.py jobNumber [hyperparameters]
# The hyperparameters are listed in order below
# 'numGen', 'syncEvery', 'actorLR', 'criticLR', \
# 'eliteFrac', 'tourneyFrac', 'mutateProb', 'mutateFrac'
#
# python -u runERLDiscrete.py 1 3 1 .01 .01 .2 .2 .9 .1

print("starting script...")

import sys
import os
import rlPulse as rlp
import spinSimulation as ss
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

np.seterr(all='raise')

print("imported libraries...")

print("Num GPUs Available: ", \
    len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", \
    len(tf.config.experimental.list_physical_devices('CPU')))

# define prefix for output files

prefix = datetime.now().strftime("%Y%m%d-%H%M%S") + f'-{int(sys.argv[1]):03}'

# make a new data directory if it doesn't exist
os.mkdir("../data/" + prefix)
os.mkdir("../data/" + prefix + "/weights")
output = open("../data/" + prefix + "/_output.txt", "a")
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

sDim = 5 # state represented by sequences of actions...
aDim = 5

numGen = int(sys.argv[2]) # how many generations to run
bufferSize = int(5e5) # size of the replay buffer
batchSize = 256 # size of batch for training, multiple of 32
popSize = 10 # size of population
polyak = .001 # polyak averaging parameter
gamma = .99 # future reward discount rate

syncEvery = int(sys.argv[3]) # how often to copy RL actor into population

actorLR = float(sys.argv[4])
criticLR = float(sys.argv[5])
lstmLayers = 1
denseLayers = 2
lstmUnits = 64
denseUnits = 256

eliteFrac = float(sys.argv[6])
tourneyFrac = float(sys.argv[7])
mutateProb = float(sys.argv[8])
mutateFrac = float(sys.argv[9])

# save hyperparameters to output file

hyperparameters = ['numGen', 'syncEvery', 'actorLR', 'criticLR', \
    'eliteFrac', 'tourneyFrac', 'mutateProb', 'mutateFrac']

hyperparamList = "\n".join([i+": "+j for i,j in \
    zip(hyperparameters, sys.argv[2:])])
print(hyperparamList)
output.write(hyperparamList)
output.write("\n" + "="*20 + "\n"*4)
output.flush()

# define algorithm objects

env = rlp.Environment(N, dim, coupling, delta, sDim, HWHH0, X, Y,\
    type='discrete')

actor = rlp.Actor(sDim,aDim, actorLR, type='discrete')
critic = rlp.Critic(sDim, aDim, gamma, criticLR, type='V')
actor.createNetwork(lstmLayers, denseLayers, lstmUnits, denseUnits)
critic.createNetwork(lstmLayers, denseLayers, lstmUnits, denseUnits)

actorTarget = actor.copy()
criticTarget = critic.copy()

pop = rlp.Population(popSize)
pop.startPopulation(sDim=sDim, aDim=aDim, learningRate=actorLR,\
    type='discrete', lstmLayers=lstmLayers, denseLayers=denseLayers, \
    lstmUnits=lstmUnits, denseUnits=denseUnits)

replayBuffer = rlp.ReplayBuffer(bufferSize)

# write actor, critic summaries to output

actor.model.summary(print_fn=lambda x: output.write(x + "\n"))
output.write("\n"*2)
critic.model.summary(print_fn=lambda x: output.write(x + "\n"))
output.write("\n"*4)
output.flush()

# define functions to save plots periodically

def makeParamDiffPlots(paramDiff, prefix):
    diffEps = [_[0] for _ in paramDiff]
    actorDiffs = np.array([_[1] for _ in paramDiff])
    criticDiffs = np.array([_[2] for _ in paramDiff])

    numFigs = 0
    for d in range(np.shape(actorDiffs)[1]):
        plt.plot(diffEps, actorDiffs[:,d], label=f"parameter {d}")
        if ((d+1) % 6 == 0):
            # 10 lines have been added to plot, save and start again
            plt.title(f"Actor parameter MSE vs target networks (#{numFigs})")
            plt.xlabel('Generation number')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.legend()
            # plt.gcf().set_size_inches(12,8)
            plt.savefig("../data/" + prefix + \
                f"/actor_param_MSE{numFigs:02}.png")
            plt.clf()
            numFigs += 1
    plt.title(f"Actor parameter MSE vs target networks (#{numFigs})")
    plt.xlabel('Generation number')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    # plt.gcf().set_size_inches(12,8)
    plt.savefig("../data/" + prefix + f"/actor_param_MSE{numFigs:02}.png")
    plt.clf()

    numFigs = 0
    for d in range(np.shape(criticDiffs)[1]):
        plt.plot(diffEps, criticDiffs[:,d], label=f"parameter {d}")
        if ((d+1) % 6 == 0):
            # 10 lines have been added to plot, save and start again
            plt.title(f"Critic parameter MSE vs target networks (#{numFigs})")
            plt.xlabel('Generation number')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.legend()
            # plt.gcf().set_size_inches(12,8)
            plt.savefig("../data/" + prefix + \
                f"/critic_param_MSE{numFigs:02}.png")
            plt.clf()
            numFigs += 1
    plt.title(f"Critic parameter MSE vs target networks (#{numFigs})")
    plt.xlabel('Generation number')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    # plt.gcf().set_size_inches(12,8)
    plt.savefig("../data/" + prefix + f"/critic_param_MSE{numFigs:02}.png")
    plt.clf()

def makePopFitPlot(fitnessMat, prefix):
    popFitGens = [_[0] for _ in fitnessMat]
    popFits = [_[1] for _ in fitnessMat]

    for i in range(len(popFitGens)):
        g = popFitGens[i]
        plt.plot([g] * len(popFits[i]), popFits[i], '.k')
    plt.title(f"Population fitnesses by generation")
    plt.xlabel('Generation number')
    plt.ylabel('Fitness')
    # plt.yscale('log')
    plt.savefig("../data/" + prefix + f"/pop_fit.png")
    plt.clf()

def makeTestPlot(testMat, prefix):
    testGens = [_[0] for _ in testMat]
    testFits = [_[1] for _ in testMat]

    plt.plot(testGens, testFits, '.k')
    plt.title(f"Test fitnesses by generation")
    plt.xlabel('Generation number')
    plt.ylabel('Fitness')
    # plt.yscale('log')
    plt.savefig("../data/" + prefix + f"/test_fit.png")
    plt.clf()

# record test results and other outputs from run
testFile = open("../data/"+prefix+"/testResults.txt", 'a')
testFile.write("Test results\n\n")
paramDiff = []
fitnessMat = [] # generation, array of fitnesses
testMat = [] # generation, fitness from test

samples = 250

###
### ERL Algorithm
###

startTime = datetime.now()
print(f"starting ERL algorithm ({startTime})")
output.write(f"started ERL algorithm: {startTime}\n")

# # build up buffer
# while replayBuffer.size < batchSize:
#     print(f"building buffer, current size is {replayBuffer.size}")
#     actor.evaluate(env, replayBuffer)

for i in range(numGen):
    timeDelta = (datetime.now() - startTime).total_seconds()
    print("="*20 + f"\nOn generation {i} ({timeDelta/60:.01f} minutes, " + \
        f'{timeDelta/(i+1):.01f} s/generation)')
    
    # evaluate the population
    pop.evaluate(env, replayBuffer, numEval=2)
    
    # evaluate the actor with noise for replayBuffer
    f = actor.evaluate(env, replayBuffer, numEval=2)
    print(f"evaluated gradient actor,\tfitness is {f:.02f}")
    
    if i % int(np.ceil(numGen / samples)) == 0:
        # record population fitnesses
        fitnessMat.append((i, np.copy(pop.fitnesses)))
        makePopFitPlot(fitnessMat, prefix)
        
        # record test results
        print("="*20 + f"\nRecording test results (generation {i})")
        if f > np.max(pop.fitnesses):
            testActor = actor
            print(f'gradient actor has highest fitness (f={f:.02f})')
            testActorType = 'gradient'
        else:
            testInd = np.argmax(pop.fitnesses)
            testActor = pop.pop[testInd]
            print('actor in population has highest fitness '+\
                f'(f={pop.fitnesses[testInd]:.02f})')
            testActorType = f'population (synced: {pop.synced[testInd]},' + \
                f'mutated: {pop.mutated[testInd]})'
        # s, rMat = testActor.test(env)
        s, rMat, criticMat = testActor.test(env, critic)
        f = np.max(rMat)
        print(f'Fitness from test: {f:0.02f}')
        testMat.append((i, f))
        makeTestPlot(testMat, prefix)
        testFile.write(f"Test result from generation {i}, " + \
            f"actor type {testActorType}\n\n")
        testFile.write("Pulse sequence:\n")
        testFile.write(rlp.formatActions(s, type=actor.type) + "\n")
        testFile.write("Critic values from pulse sequence:\n")
        print(criticMat)
        for cInd, testVal in enumerate(criticMat):
            testFile.write(f"{cInd}: {testVal:.02f}\n")
        testFile.write("\nRewards from pulse sequence:\n")
        for rInd, testR in enumerate(rMat):
            testFile.write(f"{rInd}: {testR:.02f}\n")
        testFile.write(f'\nFitness: {f:.02f}')
        testFile.write("\n"*3)
        testFile.flush()
        
        # calculate difference between parameters for actors/critics
        paramDiff.append((i, actor.paramDiff(actorTarget), \
                                 critic.paramDiff(criticTarget)))
        makeParamDiffPlots(paramDiff, prefix)
        # save actor and critic weights
        actor.save_weights("../data/"+prefix+"/weights/actor_weights.ckpt")
        critic.save_weights("../data/"+prefix+"/weights/critic_weights.ckpt")
    
    # iterate population (select elites, mutate rest of population)
    pop.iterate(eliteFrac=eliteFrac, tourneyFrac=tourneyFrac, \
         mutateProb=mutateProb, mutateFrac=mutateFrac, generation=i)
    print("iterated population")
    
    # train critic/actor networks
    batch = replayBuffer.getSampleBatch(batchSize)
    critic.trainStep(batch)
    actor.trainStep(batch, critic)
    # update target networks
    criticTarget.copyParams(critic, polyak)
    actorTarget.copyParams(actor, polyak)
    print("trained actor/critic networks")
    
    print(f'buffer size is {replayBuffer.size}\n')
    
    if i % syncEvery == 0:
        # sync actor with population
        pop.sync(actor, generation=i)

testFile.flush()
testFile.close()

timeDelta = (datetime.now() - startTime).total_seconds() / 60
print(f"finished ERL algorithm, duration: {timeDelta:.02f} minutes")
output.write(f"finished ERL algorithm: {timeDelta:.02f} minutes\n\n")
output.write("="*50 + "\n")

output.flush()
output.close()

# results (save everything to files)

# param differences
makeParamDiffPlots(paramDiff, prefix)
# population fitnesses
makePopFitPlot(fitnessMat, prefix)
# test fitnesses
makeTestPlot(testMat, prefix)

# TODO put in other results...

print("finished running script!")
