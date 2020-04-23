#!/usr/bin/env python
#
# python runRLPulse.py learningRate numExp bufferSize batchSize ...
#                 polyak updateEvery numUpdates
#
# Outline what hyperparameters I'm specifying above...
#
# standalone script to
import sys
import os
import rlPulse as rlp
import spinSimulation as ss
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# define prefix for output files

prefix = "-".join(sys.argv[1:]) + "-" + \
            datetime.now().strftime("%Y%m%d-%H%M%S")

hyperparameters = ["Learning rate", "Num experiences", "Buffer size", \
    "Batch size", "Polyak", "Update every", "Num updates"]

# make a new data directory if it doesn't exist
os.mkdir("../data/" + prefix)
output = open("../data/" + prefix + "/output.txt", "a")
output.write("Output file for run\n\n")
output.write("\n".join([i+": "+j for i,j in zip(hyperparameters, sys.argv[1:])]))
output.write("\n" + "="*50 + "\n\n")

# initialize system parameters

N = 4
dim = 2**N

# pulse = .25e-6    # duration of pulse
# delay = 3e-6      # duration of delay
coupling = 5e3    # coupling strength
delta = 500       # chemical shift strength (for identical spins)

(x,y,z) = (ss.x, ss.y, ss.z)
(X,Y,Z) = ss.getTotalSpin(N, dim)

Hdip, Hint = ss.getAllH(N, dim, coupling, delta)
HWHH0 = ss.getHWHH0(X,Y,Z,delta)

# initialize RL algorithm hyperparameters

sDim = 3 # state represented by sequences of actions...
aDim = 3 # action = [phi, rot, time]
learningRate = float(sys.argv[1]) # learning rate for optimizer

numExp = int(sys.argv[2]) # how many experiences to "play" through and learn from
bufferSize = int(sys.argv[3]) # size of the replay buffer (i.e.
                              # how many experiences to keep in memory).
batchSize = int(sys.argv[4]) # size of batch (subset of replay buffer) to use as training
               # for actor and critic.
p = 1 # action noise parameter
polyak = float(sys.argv[5]) # polyak averaging parameter
gamma = 1 # future reward discount rate

updateAfter = bufferSize # start updating actor/critic networks after this many episodes
updateEvery = int(sys.argv[6])  # update networks every __ episodes
if updateEvery > bufferSize:
    print("updateEvery is larger than bufferSize, reconsider this...")
    raise
numUpdates = int(sys.argv[7]) # how many training updates to perform on a random subset of
               # experiences (s,a,r,s1,d)
randomizeDipolarEvery = 10

pDiff = 0

# define actors/critics

actor = rlp.Actor(sDim,aDim, learningRate)
actorTarget = rlp.Actor(sDim,aDim, learningRate)
critic = rlp.Critic(sDim,aDim,None, gamma, learningRate)
criticTarget = rlp.Critic(sDim,aDim,None, gamma, learningRate)
env = rlp.Environment(N, dim, sDim, HWHH0, X, Y)

actorTarget.setParams(actor.getParams())
criticTarget.setParams(critic.getParams())

replayBuffer = rlp.ReplayBuffer(bufferSize)

# DDPG algorithm

rMat = np.zeros((numExp,))
actorAMat = np.zeros((numExp,aDim))
aMat = np.zeros((numExp,aDim))
timeMat = np.zeros((numExp, 2)) # record length of episode so far and number of pulses
# keep track of when resets/updates happen
resetStateEps = []
updateEps = []

print("starting DDPG algorithm\n", datetime.now())

for i in range(numExp):
    if i % 100 == 0:
        print("On iteration", i)
    # randomize dipolar coupling strengths for Hint
    if i > 0 and i % randomizeDipolarEvery == 0:
        Hdip, Hint = ss.getAllH(N, dim, coupling, delta)
    
    s = env.getState()
    # get action based on current state and some level of noise
    actorA = actor.predict(env.state)
    a = rlp.clipAction(actorA + rlp.actionNoise(p))
    # evolve state based on action
    env.evolve(a, Hint)
    # get reward
    r = env.reward2()
    
    # record episode data
    aMat[i,:] = a
    actorAMat[i,:] = actorA
    rMat[i] = r
    timeMat[i,:] = [env.t, np.sum(env.state[:,2] != 0)]
    
    # get updated state, and whether it's a terminal state
    s1 = env.getState()
    d = env.isDone()
    replayBuffer.add(s,a,r,s1,d)
    
    # update noise parameter
    p += pDiff
    if p < 0:
        p = 0
        pDiff = 0
    
    # CHECK IF TERMINAL
    if d:
        env.reset()
        resetStateEps.append(i)
    # UPDATE NETWORKS
    if (i > updateAfter) and (i % updateEvery == 0):
        # print("updating actor/critic networks (episode {})".format(i))
        # reset noise parameter
        p = 1
        pDiff = (0-p)/(int(updateEvery/2))
        updateEps.append(i)
        for update in range(numUpdates):
            batch = replayBuffer.getSampleBatch(batchSize)
            # train critic
            critic.trainStep(batch, actorTarget, criticTarget)
            # train actor
            actor.trainStep(batch, critic)
            # update target networks
            criticTarget.updateParams(critic, polyak)
            actorTarget.updateParams(actor, polyak)
            
print("finished DDPG algorithm\n", datetime.now())

# results (save everything to files)

plt.hist(rMat, bins=20, color='black', label='rewards')
plt.title('Rewards histogram')
plt.legend()
plt.savefig("../data/" + prefix + "/rewards_hist.pdf")
plt.clf()

plt.plot(rMat, 'ok', label='rewards')
ymin, ymax = plt.ylim()
plt.vlines(updateEps, ymin, ymax, color='red', alpha=0.2, label='updates')
#plt.vlines(resetStateEps, ymin, ymax, color='blue', alpha=0.2, \
    # linestyles='dashed', label='state reset')
plt.title('Rewards for each episode')
plt.xlabel('Episode number')
plt.ylabel('Reward')
plt.legend()
plt.savefig("../data/" + prefix + "/rewards_episode.pdf")
plt.clf()

plt.plot(aMat[:,0], 'ok', label='phi', zorder=1)
plt.plot(actorAMat[:,0], '.b', label='phi (actor)', zorder=2)
plt.title('Phi action')
ymin, ymax = plt.ylim()
plt.vlines(updateEps, ymin, ymax, color='red', alpha=0.2, label='updates')
plt.xlabel('Episode number')
plt.ylabel('Phi action')
plt.legend()
plt.savefig("../data/" + prefix + "/action_phi.pdf")
plt.clf()

plt.plot(aMat[:,1], 'ok', label='rot')
plt.plot(actorAMat[:,1], '.b', label='rot (actor)', zorder=2)
plt.title('Rot action')
ymin, ymax = plt.ylim()
plt.vlines(updateEps, ymin, ymax, color='red', alpha=0.2, label='updates')
plt.xlabel('Episode number')
plt.ylabel('Rot action')
plt.legend()
plt.savefig("../data/" + prefix + "/action_rot.pdf")
plt.clf()

plt.plot(aMat[:,2], 'ok', label='time')
plt.plot(actorAMat[:,2], '.b', label='time (actor)', zorder=2)
plt.title('Time action')
ymin, ymax = plt.ylim()
plt.vlines(updateEps, ymin, ymax, color='red', alpha=0.2, label='updates')
plt.xlabel('Episode number')
plt.ylabel('Time action')
plt.legend()
plt.savefig("../data/" + prefix + "/action_time.pdf")
plt.clf()

plt.plot(timeMat[:,0], 'ok', label='time')
plt.title('Pulse sequence length (time)')
ymin, ymax = plt.ylim()
plt.vlines(updateEps, ymin, ymax, color='red', alpha=0.2, label='updates')
plt.xlabel('Episode number')
plt.ylabel('Pulse sequence length (s)')
plt.legend()
plt.savefig("../data/" + prefix + "/sequence_length.pdf")
plt.clf()

plt.plot(timeMat[:,1], 'ok', label='time')
plt.title('Pulse sequence length (number of pulses)')
ymin, ymax = plt.ylim()
plt.vlines(updateEps, ymin, ymax, color='red', alpha=0.2, label='updates')
plt.xlabel('Episode number')
plt.ylabel('Number of pulses')
plt.legend()
plt.savefig("../data/" + prefix + "/sequence_number.pdf")
plt.clf()

# calculate other benchmarks of run

rBuffer = np.array([_[2] for _  in replayBuffer.buffer])
indSorted = rBuffer.argsort()
for i in range(1,5):
    output.write("Highest rewards in buffer (#{})\n".format(i))
    output.write("Index in buffer: {}\n".format(indSorted[-i]))
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
    r = np.log10(1+1e-12-fMean**(20e-6/t))
    output.write(f"Reward: {r}")

# TODO also see what the last sequence was somehow...

# clean up everything

output.close()
