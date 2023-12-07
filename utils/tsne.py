import os
import random
import numpy as np
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

def tsne_analyze(test_tag, features, labels, classes, save_path, feature_num=None, do_fit = True):
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
    plt.legend(classes, loc='upper right')
    plt.savefig(os.path.join(save_path, "{}.png".format(test_tag)), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, "{}.pdf".format(test_tag)), bbox_inches='tight')
    plt.close()