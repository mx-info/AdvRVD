import networkx as nx
import numpy as np
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import warnings

warnings.filterwarnings(action="ignore")
from cogdl.models.emb.node2vec import Node2vec
from cogdl.models.emb.deepwalk import DeepWalk
from cogdl.models.emb.line import LINE
from cogdl.data import Graph
import numpy as np
import torch


# if len(pdg.nodes) > 0:
# node2vec = Node2Vec(graph, dimensions=128, walk_length=30, num_walks=200,
#                     workers=96)
# w2v = node2vec.fit(window=5, min_count=1, batch_words=4)
# key2index = w2v.wv.key_to_index
# vectors = w2v.wv.vectors
# else:
# key2index = None

# if key2index is None:
#     emb_node2vec = np.zeros_like(line_vec)
# else:
# emb_node2vec = vectors[key2index[label]]

def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input',
                        default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/pdgs/Vul',
                        help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out',
                        default='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/train/images/Vul',
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
    # try:
    pdg = graph_extraction(dot)
    nodes = dict()
    index = 0
    for node in pdg.nodes():
        nodes[node] = index
        index = index + 1
    edges = []
    for edge in pdg.edges():
        e1 = nodes.get(edge[0])
        e2 = nodes.get(edge[1])
        edges.append([e1, e2])
    if len(edges) > 0:
        edge_index = np.array(edges)
        edge_index = torch.from_numpy(edge_index)
        edge_index = (edge_index[:, 0], edge_index[:, 1])
        graph = Graph(edge_index=edge_index)
        node2vec = Node2vec(dimension=128, walk_length=20, walk_num=40, window_size=1, worker=96, iteration=10, p=1,
                            q=1)
        deepwalk = DeepWalk(dimension=128, walk_length=20, walk_num=40,
                            window_size=1, worker=96, iteration=10)
        line = LINE(dimension=128, walk_length=20, walk_num=40, negative=5, batch_size=1000, alpha=0.025, order=2)
        emb_node2vec = node2vec(graph)
        emb_deepwalk = deepwalk(graph)
        emb_line = line(graph)
    else:
        emb_node2vec = None
        emb_deepwalk = None
        emb_line = None
    # print(emb)

    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    for label, all_code in labels_dict.items():
        # code = all_code.split('code:')[1].split('\\n')[0]
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace("static void", "void")
        labels_code[label] = code
    # print(labels_code)
    # degree_cen_dict = nx.degree_centrality(pdg)
    # closeness_cen_dict = nx.closeness_centrality(pdg)
    # harmonic_cen_dict = nx.harmonic_centrality(pdg)

    # G = nx.DiGraph()
    # G.add_nodes_from(pdg.nodes())
    # G.add_edges_from(pdg.edges())
    # katz_cen_dict = nx.katz_centrality(G)
    # print(degree_cen_dict)
    # print(closeness_cen_dict)
    # print(harmonic_cen_dict)
    # print(katz_cen_dict)

    node2vec_channel = []
    deepwalk_channel = []
    line_channel = []

    for label, code in labels_code.items():
        line_vec = sentence_embedding(code)
        line_vec = np.array(line_vec)
        if emb_node2vec is None:
            vec_node2vec = torch.zeros_like(line_vec)
        else:
            vec_node2vec = emb_node2vec[nodes[label]]

        if emb_deepwalk is None:
            vec_deepwalk = torch.zeros_like(line_vec)
        else:
            vec_deepwalk = emb_deepwalk[nodes[label]]

        if emb_line is None:
            vec_line = torch.zeros_like(line_vec)
        else:
            vec_line = emb_line[nodes[label]]

        node2vec_channel.append(line_vec + vec_node2vec)
        deepwalk_channel.append(line_vec + vec_deepwalk)
        line_channel.append(line_vec + vec_line)

    return (node2vec_channel, deepwalk_channel, line_channel)


# def write_to_pkl(dot, out, existing_files):
def write_to_pkl(dot, out):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    # if dot_name in existing_files:
    #     return None
    # else:
    # print(dot_name)
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
    existing_files = glob.glob(out_path + "/*.pkl")
    existing_files = [f.split('.pkl')[0] for f in existing_files]

    # for dot in dotfiles:
    #     write_to_pkl(dot, out_path)

    pool = Pool(96)
    # pool.map(partial(write_to_pkl, out=out_path, existing_files=existing_files), dotfiles)
    pool.map(partial(write_to_pkl, out=out_path), dotfiles)

    # sent2vec_model.release_shared_mem(trained_model_path)


if __name__ == '__main__':
    # image = image_generation("/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/valid/pdgs/Vul/Bad4644.dot")

    main()
    # path = "./data/real_data"
    # save_path = "./data/outputs"
    # dataset_name = os.listdir(path)
    # for dataset in dataset_name:
    #     pathname = path + "/" + dataset
    #     for type_name in os.listdir(pathname):
    #         full_path = pathname + "/" + type_name
    #         save_dir = save_path + "/" + dataset + "/" + type_name
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         main(full_path, save_dir)

    # pathname ="./pdgs"
    # save_path = "./data/outputs"
    # for type_name in os.listdir(pathname):
    #     full_path = pathname + "/" + type_name
    #     save_dir = save_path + "/sard-2/" + type_name
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     main(full_path, save_dir)
