RED = '\033[31m'
END = '\033[0m'
ARG = "maskrcnn_23"

DATADIR = "/home/kby/mnt/hdd/coffee/PatchCore/data/"
MODELNAME = f"/home/kby/mnt/hdd/coffee/PatchCore/weights/coffee_{ARG}.pth"
DATASET = "coffee"
BACKBONE = "resnet50"
# THRESHOLD = 23        #WideResNet50
THRESHOLD = 14.44          #Resnet50
# THRESHOLD = 54          #Effib4
DIRNAME = "/home/kby/mnt/hdd/coffee/PatchCore/coffee"
TRAINDIR = f"{DIRNAME}/train_"
TESTDIR = f"{DIRNAME}/test_"
RESULTDIR = f"/home/kby/mnt/hdd/coffee/PatchCore/coffee/result/"
HEADER = ["FileName", "Score", "Flag"]
