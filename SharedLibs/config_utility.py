import json

def writeConfig(cfg_file=None, settings=None):
    with open(cfg_file,"w") as f:
        json.dump(settings,fp=f)


def readConfig(cfg_file=None)->dict:
    ret = None
    with open(cfg_file, 'r') as f:
        ret = json.load(f)
    return ret