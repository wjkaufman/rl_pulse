# Create job files for all permutations of parameters
# Parameters are:
# learningRate bufferSize polyak updateEvery numExp

import shutil
import sys

learningRates = [.01, .001, .0001]
bufferSizes = [100, 200, 400]
polyaks = [.2, .8]
updateEverys = [50, 100, 200]
numExps = [1000, 2000]

i = 0

for lr in learningRates:
    for bs in bufferSizes:
        for p in polyaks:
            for ue in updateEverys:
                for ne in numExps:
                    # copy job template
                    shutil.copyfile(sys.argv[1], f"job{i:05}.pbs")
                    f = open(f"job{i:05}.pbs", 'a')
                    call = "python runRLPulse.py " + str(lr) + " " + \
                        str(bs) + " " + str(p) + " " + str(ue) + " " + str(ne)
                    f.write(call)
                    f.write("\n\nexit 0\n")
                    f.close()
                    i += 1

print("Done!")
