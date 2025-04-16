ARG = ""

DATADIR = "/home/kby/mnt/hdd/coffee/PatchCore/data/"
MODELNAME = f"/home/kby/mnt/hdd/coffee/PatchCore/weights/coffee_{ARG}.pth"
DATASET = "coffee"
BACKBONE = "wide_resnet50_2"
THRESHOLD = 23
DIRNAME = "/home/kby/mnt/hdd/coffee/PatchCore/coffee"
TRAINDIR = f"{DIRNAME}/train_{ARG}"
TESTDIR = f"{DIRNAME}/test_{ARG}"
RESULTDIR = f"/home/kby/mnt/hdd/coffee/PatchCore/coffee/result/{ARG}"
HEADER = ["FileName", "Score", "Flag"]
