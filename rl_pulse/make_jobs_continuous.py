# Create job files for all permutations of parameters
# see runRLPulse.py for what the function call should look like
#
# numExp, lstmLayers, fcLayers, lstmUnits, fcUnits
# actorLR, criticLR, polyak, gamma
# bufferSize, batchSize, updateAfter, updateEvery

import shutil
import sys

stddevs = [1e-2, 1e-3]
epsilons = [.1, .2, .3, .5]
c1s = [10, 1e2, 1e3]
penalties = [0, 1e-4, 1e-3]

i = 0

for a in stddevs:
    for b in epsilons:
        for c in c1s:
            for d in penalties:
                # copy job template
                shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                jobFile = open(f"job{i:03}.pbs", 'a')
                # create function call
                call = (f"python -u continuous_control.py {a} {b} {c} {d}")
                print(call)
                jobFile.write("echo " + call + "\n")
                jobFile.write(call)
                jobFile.write("\n\nexit 0\n")
                jobFile.close()
                i += 1

print("Done!")
