# Create job files for all permutations of parameters
# the python call should be like:
# python runRLPulse.py learningRate numExp bufferSize batchSize ...
#                 polyak updateEvery numUpdates
# (see runRLPulse.py to make sure)

import shutil
import sys

learningRates = [.001]
numExps = [5000, 20000]
bufferSizes = [500, 1000, 2000]
batchSizes = [100, 200, 500]
polyaks = [.75]
# updateEverys = [250, 500, 1000]
numUpdates = [4]

i = 0

for a in numExps:
    for b in bufferSizes:
        for c in batchSizes:
            for d in [.5,1]:
                for e in numUpdates:
                    # define updateEvery based on buffer size
                    ue = int(b * d)
                    # copy job template
                    shutil.copyfile(sys.argv[1], f"job{i:05}.pbs")
                    f = open(f"job{i:05}.pbs", 'a')
                    call = "python runRLPulse.py " + str(.001) + " " + \
                        str(a) + " " + str(b) + " " + str(c) + " " + str(.75) + \
                        " " + str(ue) + " " + str(4)
                    f.write(call)
                    f.write("\n\nexit 0\n")
                    f.close()
                    i += 1

print("Done!")
