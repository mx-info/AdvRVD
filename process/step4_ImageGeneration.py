import networkx as nx
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np

from mx_cogdl.emb.node2vec import Node2vec
from mx_cogdl.emb.deepwalk import DeepWalk
from mx_cogdl.emb.line import LINE

node2vec = Node2vec(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5, p=1.0, q=1.0)
deepwalk = DeepWalk(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5)
line = LINE(dimension=128, walk_length=30, walk_num=200, batch_size=1000, negative=5, alpha=0.025, order=3)


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input',
                        default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/pdgs/No-Vul',
                        help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out',
                        default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/images/No-Vul',
                        help='The path of output.', required=False)
    parser.add_argument('-m', '--model',
                        default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/vulcnn/data_model.bin',
                        help='The path of model.', required=False)
    args = parser.parse_args()
    return args


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot):
    pdg = graph_extraction(dot)
    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    for label, all_code in labels_dict.items():
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace("static void", "void")
        labels_code[label] = code
    graph = nx.DiGraph()
    graph.add_nodes_from(pdg.nodes())
    graph.add_edges_from(pdg.edges())
    if len(pdg.nodes) > 0:
        emb1 = node2vec(graph)
        # emb2 = deepwalk(graph)
        # emb3 = line(graph)
    else:
        emb1 = None
        # emb2 = None
        # emb3 = None
    channels1 = []
    # channels2 = []
    # channels3 = []
    for label, code in labels_code.items():
        line_vec = sentence_embedding(code)
        line_vec = np.array(line_vec)
        if emb1 is None:
            emb11 = np.zeros_like(line_vec)
            # emb22 = np.zeros_like(line_vec)
            # emb33 = np.zeros_like(line_vec)
        else:
            emb11 = emb1[label]
            # emb22 = emb2[label]
            # emb33 = emb3[label]
        channels1.append(line_vec + emb11)
        # channels2.append(line_vec + emb11)
        # channels3.append(line_vec + emb11)
    return (channels1, channels1, channels1)


def write_to_pkl(dot, out):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    channels = image_generation(dot)
    if channels == None:
        return None
    else:
        out_pkl = out + dot_name + '.pkl'
        (node2vec_channel, deepwalk_channel, line_channel) = channels
        data = [node2vec_channel, deepwalk_channel, line_channel]
        with open(out_pkl, 'wb') as f:
            pickle.dump(data, f)


def main():
    args = parse_options()
    dir_name = args.input
    out_path = args.out
    trained_model_path = args.model
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(trained_model_path)

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    dotfiles = glob.glob(dir_name + '*.dot')

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pool = Pool(96)
    pool.map(partial(write_to_pkl, out=out_path), dotfiles)



if __name__ == '__main__':
    main()
