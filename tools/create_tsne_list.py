from mmcv import Config

if __name__ == "__main__":

    filepathlist = [
        # TO BE DONE
    ]

    classes = [
        # TO BE DONE
    ]

    assert len(filepathlist) == len(classes)

    # specify a save path here
    savepath = "./"

    all_infos = []
    for idx, filepath in enumerate(filepathlist):
        with open(filepath, "r") as f:
            infos = f.readlines()
        infos = [each.strip("\n") for each in infos]
        # reassign labels
        infos = [each.split("\t")[0] + "\t" + str(idx) for each in infos]
        all_infos.extend(infos)

    f = open(savepath, "w")
    for line in all_infos:
        f.write(line + "\n")
    f.close()

    savepath = savepath.replace(".txt", "_infos.txt")
    f = open(savepath, "w")
    for idx, cls in enumerate(classes):
        f.write(str(idx) + " " + cls + "\n")
    f.write("\n")
    for filepath in filepathlist:
        f.write(filepath + "\n")
    f.close()

    print("FINISH")