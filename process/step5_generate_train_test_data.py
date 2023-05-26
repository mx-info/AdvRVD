import pickle, os, glob
import argparse
import pandas as pd
from collections import Counter


def sava_data(filename, data):
    print("开始保存数据至于：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split train datasettest_data.')
    parser.add_argument('-i', '--input', default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/images/',
                        help='The path of a dir which consists of some pkl_files')
    parser.add_argument('-o', '--out', default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/',
                        help='The path of output.', required=False)
    parser.add_argument('-t', '--type', default='train.pkl', required=False)
    args = parser.parse_args()
    return args


def generate_dataframe(input_path, save_path):
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dic = []
    for type_name in os.listdir(input_path):
        dicname = input_path + type_name
        filename = glob.glob(dicname + "/*.pkl")
        for file in filename:
            data = load_data(file)
            dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"),
                "length": len(data[0]),
                "data": data,
                "label": 0 if type_name == "No-Vul" else 1})
    final_dic = pd.DataFrame(dic)
    sava_data(save_path + "data.pkl", final_dic)


def main():
    args = parse_options()
    input_path = args.input
    output_path = args.out
    type = args.type
    generate_dataframe(input_path, output_path)
    all_data = pd.read_pickle(output_path + "data.pkl")
    from sklearn.utils import shuffle
    train = shuffle(all_data, random_state=44)
    train = train.reset_index(drop=True)
    sava_data(output_path + type, train)


if __name__ == "__main__":
    main()
