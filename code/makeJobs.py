# Create job files for all permutations of parameters
# see runRLPulse.py for what the function call should look like
#
# numExp, lstmLayers, fcLayers, lstmUnits, fcUnits
# actorLR, criticLR, polyak, gamma
# bufferSize, batchSize, updateAfter, updateEvery

import shutil
import sys

lstmLayers = [1, 2]
lstmUnits = [16,64]
actorLRs = [1e-3]
updateEverys = [50]

i = 0

for a in lstmLayers:
    for b in lstmUnits:
        for c in actorLRs:
            for d in updateEverys:
                # copy job template
                shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                jobFile = open(f"job{i:03}.pbs", 'a')
                # create function call
                call = f"python -u runRLPulse.py {100000} {a} {4} {b} {b} {c} {c*10} {.01} {.9} {10000} {250} {1000} {d}"
                print(call)
                jobFile.write("echo " + call + "\n")
                jobFile.write(call)
                jobFile.write("\n\nexit 0\n")
                jobFile.close()
                i += 1

print("Done!")
