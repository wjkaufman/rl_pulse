"""
List of pulse sequences, either existing from literature,
identified by our collaborators at MIT, or identified by
our own RL algorithms.
"""

whh4 = [0, 1, 4, 0, 3, 2]

cory48 = [
    1, 3, 0, 2, 3, 0, 1, 3, 0,
    1, 3, 0, 1, 4, 0, 1, 3, 0,
    4, 2, 0, 3, 2, 0, 4, 2, 0,
    4, 2, 0, 4, 1, 0, 4, 2, 0,
    2, 3, 0, 2, 4, 0, 2, 3, 0,
    1, 4, 0, 2, 4, 0, 1, 4, 0,
    3, 2, 0, 3, 1, 0, 3, 2, 0,
    4, 1, 0, 3, 1, 0, 4, 1, 0
]

ideal6 = [3, 1, 1, 3, 2, 2]
yxx24 = [
    4, 1, 2, 3, 2, 2, 3, 2, 1, 4, 1, 1,
    3, 2, 1, 4, 1, 1, 4, 1, 2, 3, 2, 2
]
yxx48 = [
    3, 2, 2, 3, 2, 2, 4, 1, 1, 3, 2, 2,
    4, 1, 1, 4, 1, 1, 3, 2, 2, 3, 2, 2,
    4, 1, 1, 3, 2, 2, 4, 1, 1, 4, 1, 1,
    3, 2, 2, 4, 1, 1, 3, 2, 2, 4, 1, 1
]

# brute-force search
bf6 = [1, 1, 3, 1, 1, 3]
bf12 = [1, 1, 4, 1, 1, 4,
        2, 2, 4, 2, 2, 4]
bfr12 = [1, 4, 4, 1, 4, 4,
         1, 3, 3, 1, 3, 3]

# vanilla MCTS search
mcts12_1 = [0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1]
mcts12_2 = [4, 0, 4, 1, 4, 4, 3, 4, 2, 2, 2, 0]
mcts12_3 = [4, 4, 4, 1, 0, 3, 3, 0, 1, 3, 3, 4]
mcts12_4 = [3, 0, 3, 1, 3, 0, 3, 2, 1, 1, 1, 2]
mcts24 = [4, 2, 3, 4, 2, 1, 3, 2, 0, 2, 2, 3,
          4, 0, 3, 1, 2, 1, 3, 4, 1, 1, 1, 2]

# AlphaZero MCTS search
# az1 spends equal time on each axis, is cyclic,
# but has low fidelity. Why??
az1 = [
    0, 3, 3, 1, 1, 2, 3, 4, 0, 1, 3, 3,
    1, 0, 2, 2, 2, 0, 3, 3, 4, 1, 3, 3,
    2, 2, 4, 3, 1, 1, 2, 1, 4, 0, 2, 2,
    3, 2, 0, 1, 2, 1, 1, 4, 4, 0, 0, 3
]

# from 2/18 job
az12 = [  # 5.68 reward, vs 5.6 for yxx24 and 5 for yxx48
    1, 4, 2, 2, 3, 2, 2, 3, 1, 1, 4, 1
]

# from ???, probably sometime between 2/18 and 3/6
az12_1 = [  # 5.6578 reward
    3, 1, 4, 4, 2, 4, 1, 4, 4, 2, 3, 3
]

# from 3/6 job
az12_2 = [1, 3, 3, 1, 3, 3, 1, 4, 4, 1, 4, 4]

# from 3/8 job
az24 = [
    3, 3, 1, 3, 3, 1, 3, 3, 2, 3, 3, 2,
    4, 4, 2, 4, 4, 2, 4, 4, 1, 4, 4, 1
]

# from 3/13 job, unconstrained search
az24_unconstrained = [
    3, 3, 2, 0, 0, 0, 0, 4, 1, 4, 4, 2,
    4, 1, 4, 4, 3, 3, 4, 2, 1, 4, 4, 4
]

# from 2/23 job
az48 = [  # 4.3 reward apparently for strongly coupled spin system
    4, 2, 1, 1, 2, 1, 2, 0, 4, 0, 3, 1,
    3, 2, 3, 0, 0, 2, 4, 4, 2, 1, 2, 2,
    2, 4, 0, 0, 2, 3, 4, 3, 4, 0, 1, 4,
    3, 1, 3, 4, 0, 0, 4, 0, 2, 0, 0, 4
]

# from 2/26 job
az48_1 = [  # 4.4 reward vs 6.7 for cory48, 6.2 for yxx48
    3, 1, 0, 0, 0, 0, 1, 3, 4, 3, 2, 3,
    4, 0, 1, 0, 3, 0, 1, 2, 4, 4, 3, 3,
    0, 0, 1, 4, 0, 3, 0, 0, 0, 3, 1, 2,
    0, 0, 0, 4, 0, 1, 0, 0, 0, 3, 0, 0
]

# from 3/1 job
az48_2 = [
    3, 0, 1, 0, 1, 1, 3, 0, 2, 1, 3, 1,
    0, 3, 0, 0, 4, 4, 2, 2, 4, 3, 4, 3,
    1, 4, 4, 3, 0, 4, 4, 2, 4, 4, 3, 4,
    0, 4, 3, 1, 4, 4, 4, 4, 4, 0, 0, 1
]

# from 3/4 job
az48_3 = [
    2, 3, 3, 1, 3, 3, 4, 1, 1, 4, 1, 1,
    3, 1, 1, 4, 1, 1, 4, 1, 4, 1, 1, 4,
    2, 2, 3, 3, 2, 3, 3, 4, 1, 4, 4, 1,
    4, 1, 1, 4, 1, 1, 4, 2, 2, 4, 2, 2
]

# from 3/11 job, designed to be robust to
# rotation and phase transient error, but
# messed up magnitudes and it's _really_
# robust to phase transients!
az48_robust = [
    2, 2, 4, 4, 2, 4, 2, 2, 3, 2, 2, 3,
    1, 2, 3, 2, 2, 3, 1, 1, 1, 4, 1, 1,
    4, 4, 1, 1, 4, 1, 4, 2, 4, 4, 2, 4,
    1, 3, 1, 1, 1, 3, 1, 4, 1, 4, 1, 4
]

# from 3/14 job, unconstrained search
az48_unconstrained = [
    1, 3, 4, 4, 3, 2, 0, 3, 4, 3, 1, 2,
    3, 4, 0, 0, 1, 2, 4, 0, 1, 4, 2, 0,
    3, 4, 2, 4, 2, 3, 2, 3, 3, 4, 3, 2,
    0, 3, 1, 4, 3, 4, 3, 1, 1, 2, 2, 1
]

'''
different length pulse sequences
from 3/6-3/8
'''
az12_3 = [1, 3, 3, 1, 3, 3, 1, 4, 4, 1, 4, 4]

az24_2 = [
    3, 3, 1, 3, 3, 1, 3, 3, 2, 3, 3, 2,
    4, 4, 2, 4, 4, 2, 4, 4, 1, 4, 4, 1
]

az36 = [
    2, 2, 4, 2, 2, 4, 1, 1, 4, 1, 1, 4,
    4, 2, 2, 4, 2, 2, 4, 2, 3, 2, 2, 3,
    2, 1, 1, 4, 1, 1, 4, 4, 1, 4, 4, 1
]

az48_4 = [
    3, 3, 1, 1, 1, 3, 3, 3, 1, 3, 3, 1,
    3, 3, 1, 3, 3, 1, 3, 3, 2, 3, 3, 2,
    1, 4, 1, 1, 4, 1, 4, 1, 3, 3, 3, 1,
    3, 4, 1, 1, 1, 4, 4, 1, 4, 4, 1, 4
]

az96 = [
    2, 4, 2, 2, 4, 2, 2, 3, 3, 1, 3, 3,
    1, 3, 2, 3, 3, 2, 4, 4, 2, 4, 4, 2,
    1, 1, 3, 1, 1, 3, 4, 1, 1, 4, 1, 1,
    4, 2, 2, 4, 2, 2, 2, 4, 1, 1, 1, 4,
    4, 4, 1, 4, 4, 1, 1, 1, 4, 1, 4, 1,
    1, 1, 4, 1, 1, 4, 1, 3, 1, 1, 3, 1,
    2, 2, 4, 4, 2, 4, 4, 1, 4, 1, 1, 4,
    3, 1, 1, 3, 1, 1, 3, 4, 2, 2, 4, 2
]

'''
unconstrained search with limited CPUs
(wanted to get results quickly)
'''
# from 3/15
az12_unconstrained = [1, 4, 3, 2, 4, 4, 1, 2, 3, 2, 1, 3]
az24_unconstrained_2 = [
    0, 3, 1, 0, 0, 0, 0, 4, 2, 0, 2, 0,
    4, 0, 4, 2, 2, 0, 4, 2, 0, 0, 0, 4
]
# 3/16
az36_unconstrained = [
    2, 2, 4, 1, 0, 0, 1, 3, 1, 4, 4, 0,
    1, 0, 4, 0, 0, 4, 4, 4, 3, 4, 1, 4,
    3, 4, 2, 3, 4, 0, 2, 0, 4, 2, 1, 1
]
az48_unconstrained_2 = [
    1, 0, 2, 3, 0, 4, 4, 4, 4, 0, 1, 4,
    4, 2, 1, 1, 1, 4, 4, 4, 0, 2, 1, 2,
    0, 1, 3, 4, 0, 3, 0, 2, 1, 1, 2, 2,
    3, 0, 2, 1, 4, 1, 0, 0, 1, 3, 3, 3
]


'''
2021-04-03
Re-running 48-pulse search with refocusing every
6tau constraint, will try 6tau, 12tau, etc.
'''

az48_5 = [
    2, 3, 3, 2, 3, 3, 1, 3, 3, 1, 3, 3,
    1, 4, 1, 1, 4, 1, 3, 1, 3, 3, 1, 3,
    4, 2, 4, 4, 2, 4, 2, 2, 3, 2, 2, 3,
    4, 1, 4, 4, 1, 4, 2, 2, 4, 2, 2, 4
]


'''
2021-04-19
Pulse sequences from AHT0/6tau constrained search, no errors
'''

az_no_err_12 = [
    1, 4, 4, 1, 4, 4, 4, 1, 1, 4, 1, 1
]

az_no_err_24 = [
    4, 4, 2, 4, 4, 2, 4, 4, 1, 1, 4, 1,
    4, 4, 1, 4, 4, 1, 1, 1, 4, 4, 1, 4
]

az_no_err_48 = [
    4, 2, 4, 2, 4, 2, 3, 1, 3, 1, 1, 3,
    3, 1, 1, 3, 1, 1, 0, 2, 4, 2, 2, 4,
    2, 2, 4, 2, 4, 2, 4, 4, 2, 2, 2, 4,
    0, 4, 4, 2, 4, 4, 1, 3, 2, 3, 2, 3
]

'''
2021-04-28
Pulse sequences from AHT0/6tau, rotation errors
'''

az_rot_err_12 = [
    0, 1, 1, 3, 1, 1, 0, 2, 2, 4, 2, 2
]

az_rot_err_24 = [
    4, 1, 3, 3, 1, 3, 3, 2, 3, 3, 2, 3,
    3, 2, 4, 4, 2, 4, 4, 1, 4, 4, 1, 4
]

az_rot_err_48 = [
    4, 4, 2, 2, 2, 4, 2, 3, 2, 2, 3, 2,
    4, 2, 4, 2, 2, 4, 3, 2, 4, 4, 2, 4,
    4, 1, 3, 3, 1, 3, 3, 2, 2, 3, 2, 2,
    4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2
]

'''
2021-05-02
Pulse sequences from AHT0/6tau, all errors
'''

az_all_err_12 = [
    4, 2, 3, 3, 2, 3, 3, 2, 4, 4, 2, 4
]
az_all_err_24 = [
    4, 4, 2, 4, 4, 2, 3, 2, 3, 3, 2, 3,
    1, 3, 1, 1, 3, 1, 4, 4, 1, 4, 4, 1
]
az_all_err_48 = [
    1, 1, 4, 1, 1, 4, 2, 4, 4, 2, 4, 4,
    2, 4, 4, 2, 4, 4, 3, 3, 3, 2, 3, 3,
    2, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2,
    1, 4, 4, 1, 4, 4, 4, 2, 2, 4, 2, 2
]
