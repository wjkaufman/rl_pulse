# Create job files for all permutations of parameters
# see runRLPulse.py for what the function call should look like

import shutil
import sys

lr = 0.001
numExps = [10000, 20000]
bufferSizes = [1000, 2000]
batchSizes = [100, 200]
polyaks = [1e-3, 1e-5]
ue = 5
# updateEverys = [250, 500, 1000]

i = 0

for a in numExps:
    for b in bufferSizes:
        for c in batchSizes:
            for d in polyaks:
                # copy job template
                shutil.copyfile(sys.argv[1], f"job{i:05}.pbs")
                f = open(f"job{i:05}.pbs", 'a')
                # create function call
                call = f"python runRLPulse.py {lr} {a} {b} {c} {d} {ue}"
                f.write(call)
                f.write("\n\nexit 0\n")
                f.close()
                i += 1

print("Done!")
