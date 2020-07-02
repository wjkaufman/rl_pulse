# Create job files for all permutations of parameters
# see runERL.py for arguments for script
#
# jubNum,
# 'numGen', 'syncEvery', 'actorLR', 'criticLR', \
# 'lstmLayers', 'denseLayers', 'lstmUnits', 'denseUnits', \
# 'normalizationType', \

import shutil
import sys

actorLRs = [1e-5]
criticMult = [1, 10]
denseLayers = [2, 8]
lstmUnits = [128]
denseUnits = [64]
normalizationTypes = ['layer', 'batch']
# eliteFracs = [.2]
# mutateProbs = [.9]
# mutateFracs = [.1]

i = 0
for a in actorLRs:
    for b in criticMult:
        for c in denseLayers:
            for d in lstmUnits:
                for e in denseUnits:
                    for f in normalizationTypes:
                        # copy job template
                        shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                        jobFile = open(f"job{i:03}.pbs", 'a')
                        # create function call
                        call = f"python -u runERLDiscrete.py {i:02} {500:0.0f} "
                        call += f"{5} {a:.0e} {b*a:.0e} {1} {c} {d} {e} {f}"
                        print(call)
                        jobFile.write("echo " + call + "\n")
                        jobFile.write(call)
                        jobFile.write("\n\nexit 0\n")
                        jobFile.close()
                        i += 1

print("Done!")
