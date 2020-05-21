# Create job files for all permutations of parameters
# see runERL.py for arguments for script
#
# jubNum, 'numGen', 'syncEvery', 'actorLR', 'criticLR', \
# 'eliteFrac', 'tourneyFrac', 'mutateProb', 'mutateFrac'

import shutil
import sys

actorLRs = [.01, .001]         # 2
criticLRs = [.01]      # 3
eliteFracs = [.2]           # 1
mutateProbs = [.25]       # 1
mutateFracs = [.1]        # 1

i = 0

for a in actorLRs:
    for b in criticLRs:
        for c in eliteFracs:
            for d in mutateProbs:
                for e in mutateFracs:
                    # copy job template
                    shutil.copyfile(sys.argv[1], f"job{i:03}.pbs")
                    jobFile = open(f"job{i:03}.pbs", 'a')
                    # create function call
                    call = f"python -u runERL.py {i:02} {2e3:0.0f} {5} "
                    call += f"{a} {b} {c} {.2} {d} {e}"
                    print(call)
                    jobFile.write("echo " + call + "\n")
                    jobFile.write(call)
                    jobFile.write("\n\nexit 0\n")
                    jobFile.close()
                    i += 1

print("Done!")
