'''
tsne visualization
'''

import os
import random
import numpy as np
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import shutil

import torch

from data.dataset_entry import get_test_loader
from models.model_entry import model_entry
from utils.log import setup_logger
from utils.parser import parse_cfg
from utils.init_config import init_config
from utils.checkpoint import load_checkpointpath

from tools.inference import inference

def tsne_analyze(features, labels, classes, save_path, feature_num=None, do_fit = True):
    
    test_tag = config.common.test_tag
    if do_fit:
        print(f">>> t-SNE fitting")
        embeddings = TSNE(n_jobs=4).fit_transform(features)
        print(f"<<< fitting over")
        np.save(os.path.join(save_path,'{}_feats.npy'.format(test_tag)), embeddings)
    else:
        embeddings=np.load(os.path.join(save_path,'{}_feats.npy'.format(test_tag)))
        labels=np.load(os.path.join(save_path, '{}_labels.npy'.format(test_tag)))
    index = [i for i in range(len(embeddings))]
    random.shuffle(index)
    embeddings = np.array([embeddings[index[i]] for i in range(len(index))])
    labels = [labels[index[i]] for i in range(len(index))]
    if feature_num is not None:
        embeddings = embeddings[:feature_num]
        labels = labels[:feature_num]

    print(f">>> draw image begin")
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.figure(figsize=(5,5))
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
     
    num_classes =  len(set(labels))
    print('num_classes', num_classes)
    for i, lab in enumerate(list(range(num_classes))):
        if i < 20:
            color = plt.cm.tab20(i)
        elif i<40:
            color = plt.cm.tab20b(i-20)
        else:
            color = plt.cm.tab20c(i-40)
        class_index = [j for j,v in enumerate(labels) if v == lab]
        plt.scatter(vis_x[class_index], vis_y[class_index], color = color, alpha=1, s=8, marker='o')

    plt.xticks([])
    plt.yticks([])
    # plt.legend(classes, loc='upper right')
    plt.savefig(os.path.join(save_path, "{}.png".format(test_tag)), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, "{}.pdf".format(test_tag)), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    arg = parse_cfg()
    config = init_config(arg, makedirs=False)

    logger = setup_logger(config, test=True)
    # savepath
    tsne_path = os.path.join(config.common.workspace, "tsne")
    os.makedirs(tsne_path, exist_ok=True)
    # set device (single GPU now)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    assert len(arg.gpu) == 1, "Single GPU now"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu[0]) # single gpu here
    device = torch.device("cuda:{}".format(str(arg.gpu[0])))

    # dataloader
    test_loader = get_test_loader(config, logger, enable_labelmap=False)
    # build model
    model = model_entry(config, logger)
    model = model.to(device)
    model.eval()
    # load best ckpt
    load_checkpointpath(config, logger, model, testmode=True, resume_best=True)


    logger.info("Extracting features for TSNE.")

    all_feats = []
    all_labels = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            imgs, labels = batch["imgs"], batch["labels"]
                    
            imgs = imgs.to(device)
            batchsize = imgs.shape[0]
            imgs = imgs.reshape((-1, 3, imgs.size(-2), imgs.size(-1)))
            labels = labels.reshape((-1)).cpu().tolist()
            output = model(imgs)
            feats = output["feats"]
            all_feats.append(feats.cpu())
            all_labels.extend(labels)
        all_feats = torch.cat(all_feats).numpy()
    
    logger.info("TSNE analyzing")
    # TODO: change classes to use config.data.CLASSES
    print(len(set(list(all_labels))))
    tsne_analyze(
        all_feats,
        all_labels,
        classes=[str(i) for i in range(len(set(list(all_labels))))],
        feature_num=4000,
        save_path=tsne_path,
        do_fit=True
    )

    test_meta = config.data.test_meta[0]
    test_tag = config.common.test_tag
    meta_infopath = test_meta.replace(".txt", "_infos.txt")
    assert os.path.exists(meta_infopath)
    shutil.copy(meta_infopath, os.path.join(tsne_path, "{}_infos.txt".format(test_tag)))


    logger.info("FINISH")