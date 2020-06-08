# Create job files for all permutations of parameters
# see runERL.py for arguments for script
#
# jubNum,
# 'numGen', 'syncEvery', 'actorLR', 'criticLR', \
# 'lstmLayers', 'denseLayers', 'lstmUnits', 'denseUnits', \
#
# currently not used:
# 'eliteFrac', 'tourneyFrac', 'mutateProb', 'mutateFrac'

import shutil
import sys

actorLRs = [1e-4, 1e-5]
criticMult = [1, 10]
denseLayers = [2,4]
lstmUnits = [32, 64]
denseUnits = [32, 64]
# eliteFracs = [.2]
# mutateProbs = [.9]
# mutateFracs = [.1]

i = 0
for a in actorLRs:
    for b in criticMult:
        for c in denseLayers:
            for d in lstmUnits:
                for e in denseUnits:
                    # copy job template
                    shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                    jobFile = open(f"job{i:03}.pbs", 'a')
                    # create function call
                    call = f"python -u runERLDiscrete.py {i:02} {1e3:0.0f} {5} "
                    call += f"{a} {b*a} {1} {c} {d} {e}"
                    print(call)
                    jobFile.write("echo " + call + "\n")
                    jobFile.write(call)
                    jobFile.write("\n\nexit 0\n")
                    jobFile.close()
                    i += 1

print("Done!")
