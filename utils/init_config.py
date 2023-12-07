import os
from mmcv import Config
import datetime

def init_config(arg, makedirs=True):
    assert os.path.exists(arg.config), "Config {} do not exist".format(arg.config)
    config = Config.fromfile(arg.config)

    config.common.exp_tag = arg.exp_tag
    
    config.common.timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    config.common.checkpointpath = os.path.join("./{}".format(config.common.exp_tag), "checkpoints")
    config.common.logpath = os.path.join("./{}".format(config.common.exp_tag), "logs")
    config.common.tblogpath = os.path.join("./{}".format(config.common.exp_tag), "tensorlogs")
    config.common.predpath = os.path.join("./{}".format(config.common.exp_tag), "preds")
    config.common.workspace = os.path.join("./{}".format(config.common.exp_tag))
    config.common.test_tag = arg.test_tag
    os.makedirs(config.common.logpath, exist_ok=True)
    if not makedirs:
        return config
    os.makedirs(config.common.checkpointpath, exist_ok=True)
    os.makedirs(config.common.tblogpath, exist_ok=True)
    os.makedirs(config.common.predpath, exist_ok=True)

    return config
