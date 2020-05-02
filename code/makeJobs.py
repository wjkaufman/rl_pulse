# Create job files for all permutations of parameters
# see runRLPulse.py for what the function call should look like

import shutil
import sys

learningRates = [1e-3, 1e-5]
numExps = [25000]
bufferSizes = [5000]
batchSizes = [500]
polyaks = [.01]
ue = 10
LSTMs = [4]
hiddens = [16]

i = 0

for z in learningRates:
    for a in numExps:
        for b in bufferSizes:
            for c in batchSizes:
                for d in polyaks:
                    for e in LSTMs:
                        for f in hiddens:
                            # copy job template
                            shutil.copyfile(sys.argv[1], f"job{i:04}.pbs")
                            jobFile = open(f"job{i:04}.pbs", 'a')
                            # create function call
                            call = f"python -u runRLPulse.py {z} {a} {b} {c} {d} {ue} {e} {f} {e} {f}"
                            print(call)
                            jobFile.write("echo " + call + "\n")
                            jobFile.write(call)
                            jobFile.write("\n\nexit 0\n")
                            jobFile.close()
                            i += 1

print("Done!")
