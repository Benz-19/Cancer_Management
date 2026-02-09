import os

def reset():
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    os.environ["OMP_NUM_THREADS"] = "3"
