# Create job files for all permutations of parameters
# see runERL.py for arguments for script
#
# jubNum, 'numGen', 'syncEvery', 'actorLR', 'criticLR', \
# 'eliteFrac', 'tourneyFrac', 'mutateProb', 'mutateFrac'

import shutil
import sys

actorLRs = [1e-4]
criticMult = [1, 10]
eliteFracs = [.2]
mutateProbs = [.9]
mutateFracs = [.1]

i = 10

for a in actorLRs:
    for b in criticMult:
        for c in eliteFracs:
            for d in mutateProbs:
                for e in mutateFracs:
                    # copy job template
                    shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                    jobFile = open(f"job{i:03}.pbs", 'a')
                    # create function call
                    call = f"python -u runERL.py {i:02} {5e3:0.0f} {5} "
                    call += f"{a} {b*a} {c} {.2} {d} {e}"
                    print(call)
                    jobFile.write("echo " + call + "\n")
                    jobFile.write(call)
                    jobFile.write("\n\nexit 0\n")
                    jobFile.close()
                    i += 1

print("Done!")
