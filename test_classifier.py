import os
import torch

from utils.checkpoint import load_checkpointpath

from utils.log import setup_logger
from utils.parser import parse_cfg
from utils.init_config import init_config
from models.model_entry import model_entry
from data.dataset_entry import get_test_loader
from tools.inference import inference


def main():
    arg = parse_cfg()
    config = init_config(arg)

    logger = setup_logger(config, test=True)

    # set device (single GPU now)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    assert len(arg.gpu) == 1, "Single GPU now"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu[0]) # single gpu here
    device = torch.device("cuda:{}".format(str(arg.gpu[0])))

    test_loader = get_test_loader(config, logger)

    # build model
    model = model_entry(config, logger)
    model = model.to(device)

    logger.info("\n----------------- START TESTING -----------------")
    with torch.no_grad():
        inference(model, config, device, test_loader, logger, ckptpath=arg.ckpt_path)

if __name__ == "__main__":
    main()