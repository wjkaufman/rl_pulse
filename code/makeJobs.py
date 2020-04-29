# Create job files for all permutations of parameters
# see runRLPulse.py for what the function call should look like

import shutil
import sys

lr = 0.001
numExps = [25000]
bufferSizes = [5000]
batchSizes = [500, 1000]
polyaks = [1e-3]
ue = 5
LSTMs = [1, 4]
hiddens = [2, 8]

i = 0

for a in numExps:
    for b in bufferSizes:
        for c in batchSizes:
            for d in polyaks:
                for e in LSTMs:
                    for f in hiddens:
                        # copy job template
                        shutil.copyfile(sys.argv[1], f"job{i:05}.pbs")
                        jobFile = open(f"job{i:05}.pbs", 'a')
                        # create function call
                        call = f"python runRLPulse.py {lr} {a} {b} {c} {d} {ue} {e} {f} {e} {f}"
                        print(call)
                        jobFile.write("echo " + call + "\n")
                        jobFile.write(call)
                        jobFile.write("\n\nexit 0\n")
                        jobFile.close()
                        i += 1

print("Done!")
