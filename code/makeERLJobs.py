# Create job files for all permutations of parameters
# see runERL.py for arguments for script
#
# numGen, bufferSize, batchSize, popSize, polyak, gamma, syncEvery,
# actorLR, criticLR, lstmLayers, fcLayers, lstmUnits, fcUnits

import shutil
import sys

numGens = [1000]
actorLRs = [1e-2, 1e-3]
lstmLayers=[1,2]
fcLayers = [2,4,8]
units = [16, 32] # for both LSTM and dense layers

i = 0

for b in actorLRs:
    for c in lstmLayers:
        for d in fcLayers:
            for e in units:
                # copy job template
                shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                jobFile = open(f"job{i:03}.pbs", 'a')
                # create function call
                call = f"python -u runERL.py {i:02} {1e3:0.0f} {1e5:0.0f} {1000} {10} {.01} {.99} {5} {b} {b*10} {c} {d} {e} {e}"
                print(call)
                jobFile.write("echo " + call + "\n")
                jobFile.write(call)
                jobFile.write("\n\nexit 0\n")
                jobFile.close()
                i += 1

print("Done!")
