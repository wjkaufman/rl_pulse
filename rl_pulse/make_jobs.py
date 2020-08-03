# Create job files for all permutations of parameters
# see runRLPulse.py for what the function call should look like
#
# numExp, lstmLayers, fcLayers, lstmUnits, fcUnits
# actorLR, criticLR, polyak, gamma
# bufferSize, batchSize, updateAfter, updateEvery

import shutil
import sys

lstmLayers = [1,2]
denseLayers = [4,8]
# units = [64] # for both LSTM and dense layers
gamma = [0.5, 0.9]
batchSizes = [250, 1000]
updateEverys = [20, 40]

i = 0

for a in lstmLayers:
    for b in denseLayers:
        for c in batchSizes:
            for d in gamma:
                for e in updateEverys:
                    # copy job template
                    shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                    jobFile = open(f"job{i:03}.pbs", 'a')
                    # create function call
                    call = f"python -u runRLPulse.py {100000} {a} {b} {64} {64} {1e-4} {1e-3} {.01} {d} {10000} {c} {1000} {e}"
                    print(call)
                    jobFile.write("echo " + call + "\n")
                    jobFile.write(call)
                    jobFile.write("\n\nexit 0\n")
                    jobFile.close()
                    i += 1

print("Done!")
