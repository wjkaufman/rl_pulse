#!/usr/bin/env python
#
# python -u runERL.py jobNumber [hyperparameters]
# The hyperparameters are listed in order below
# 'numGen', 'syncEvery', 'actorLR', 'criticLR', \
# 'eliteFrac', 'tourneyFrac', 'mutateProb', 'mutateFrac'
#
# python -u runERL.py 1 3 1 .1 .1 .2 .2 .25 .1

print("starting runRLPulse script...")

import sys
import os
import rlPulse as rlp
import spinSimulation as ss
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

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

sDim = 3 # state represented by sequences of actions...
aDim = 3 # action = [phi, rot, time]

numGen = int(sys.argv[2]) # how many generations to run
bufferSize = int(1e5) # size of the replay buffer
batchSize = 1024 # size of batch for training, multiple of 32
popSize = 10 # size of population
polyak = .01 # polyak averaging parameter
gamma = .99 # future reward discount rate

syncEvery = int(sys.argv[3]) # how often to copy RL actor into population

p = .05

actorLR = float(sys.argv[4])
criticLR = float(sys.argv[5])
lstmLayers = 1
fcLayers = 8
lstmUnits = 64
fcUnits = 32

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

env = rlp.Environment(N, dim, coupling, delta, sDim, HWHH0, X, Y)
noiseProcess = rlp.NoiseProcess(p)

actor = rlp.Actor(sDim,aDim, actorLR)
critic = rlp.Critic(sDim, aDim, gamma, criticLR)
actor.createNetwork(lstmLayers, fcLayers, lstmUnits, fcUnits)
critic.createNetwork(lstmLayers, fcLayers, lstmUnits, fcUnits)

actorTarget = actor.copy()
criticTarget = critic.copy()

pop = rlp.Population(popSize)
pop.startPopulation(sDim, aDim, actorLR, \
    lstmLayers, fcLayers, lstmUnits, fcUnits)

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

###
### ERL Algorithm
###

samples = 250

startTime = datetime.now()
print(f"starting ERL algorithm ({startTime})")
output.write(f"started ERL algorithm: {startTime}\n")

# build up buffer
while replayBuffer.size < batchSize:
    print(f"building buffer, current size is {replayBuffer.size}")
    actor.evaluate(env, replayBuffer, noiseProcess)

for i in range(numGen):
    timeDelta = (datetime.now() - startTime).total_seconds() / 60
    print("="*20 + f"\nOn generation {i} ({timeDelta:.01f} minutes)")
    
    # evaluate and iterate the population
    pop.evaluate(env, replayBuffer, None, numEval=2)
    if i % int(np.ceil(numGen / samples)) == 0:
        fitnessMat.append((i, np.copy(pop.fitnesses)))
        makePopFitPlot(fitnessMat, prefix)
    pop.iterate(eliteFrac=eliteFrac, tourneyFrac=tourneyFrac, \
         mutateProb=mutateProb, mutateFrac=mutateFrac)
    print("iterated population")
    
    # evaluate the actor
    f = actor.evaluate(env, replayBuffer, noiseProcess)
    print(f"evaluated the actor,\tfitness is {f:.02f}")
    
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
    
    if i % int(np.ceil(numGen / samples)) == 0:
        print("="*20 + f"\nRecording test results (generation {i})")
        s, rMat = actor.test(env)
        f = np.max(rMat)
        # record results from the test
        print(f'Fitness from test: {f:0.02f}')
        testMat.append((i, f))
        makeTestPlot(testMat, prefix)
        testFile.write(f"Test result from generation {i}\n\n")
        testFile.write("Chosen pulse sequence:\n")
        testFile.write(rlp.formatAction(s) + "\n")
        testFile.write("Rewards from the pulse sequence:\n")
        for testR in rMat:
            testFile.write(f"{testR:.02f}, ")
        testFile.write(f'\nFitness: {f:.02f}')
        testFile.write("\n"*3)
        testFile.flush()
    
    print(f'buffer size is {replayBuffer.size}\n')
    
    if i % syncEvery == 0:
        # sync actor with population
        pop.sync(actor)
    
    if i % int(np.ceil(numGen/samples)) == 0:
        # calculate difference between parameters for actors/critics
        paramDiff.append((i, actor.paramDiff(actorTarget), \
                                 critic.paramDiff(criticTarget)))
        makeParamDiffPlots(paramDiff, prefix)
        # save actor and critic weights
        actor.save_weights("../data/"+prefix+"/weights/actor_weights.ckpt")
        critic.save_weights("../data/"+prefix+"/weights/critic_weights.ckpt")

timeDelta = (datetime.now() - startTime).total_seconds() / 60
print(f"finished ERL algorithm, duration: {timeDelta:.02f} minutes")
output.write(f"finished ERL algorithm: {timeDelta:.02f} minutes\n\n")
output.write("="*50 + "\n")

testFile.flush()
testFile.close()

# results (save everything to files)

# param differences
makeParamDiffPlots(paramDiff, prefix)
# population fitnesses
makePopFitPlot(fitnessMat, prefix)
# test fitnesses
makeTestPlot(testMat, prefix)

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
