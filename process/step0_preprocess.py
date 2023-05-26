import pandas as pd


def d2a():
    train = pd.read_csv(
        '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_train.csv',
        encoding='utf-8')
    test = pd.read_csv(
        '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_test.csv',
        encoding='utf-8')
    dev = pd.read_csv(
        '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_dev.csv',
        encoding='utf-8')

    # train
    for i, file in enumerate(train.values):
        file_name = file[0]
        label = file[1]
        code_text = file[2]
        if label == 1:
            path = "/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/Vul/Bad" + str(
                file_name) + ".c"
        elif label == 0:
            path = "/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/No-Vul/Good" + str(
                file_name) + ".c"
        else:
            print("path error")
        with open(path, 'w') as f:
            f.write(code_text)
            f.flush()

    # dev
    for i, file in enumerate(dev.values):
        file_name = file[0]
        label = file[1]
        code_text = file[2]
        if label == 1:
            path = "/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/valid/Vul/Bad" + str(
                file_name) + ".c"
        elif label == 0:
            path = "/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/valid/No-Vul/Good" + str(
                file_name) + ".c"
        else:
            print("path error")
        with open(path, 'w') as f:
            f.write(code_text)
            f.flush()


if __name__ == '__main__':
    d2a()
