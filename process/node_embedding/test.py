import networkx as nx
import sent2vec
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np

from node2vec import Node2Vec
from ge import DeepWalk


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    # global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(
        '/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/vulcnn/data_model.bin')
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot):
    # try:
    pdg = graph_extraction(dot)

    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    for label, all_code in labels_dict.items():
        # code = all_code.split('code:')[1].split('\\n')[0]
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace("static void", "void")
        labels_code[label] = code

    G = nx.DiGraph()
    G.add_nodes_from(pdg.nodes())
    G.add_edges_from(pdg.edges())

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200,
                        workers=96)  # Use temp_folder for big graphs


    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)  # init model
    model.train(window_size=5, iter=3)  # train model
    embeddings = model.get_embeddings()  # get embedding vectors


    # Embed nodes
    w2v = node2vec.fit(window=5, min_count=1, batch_words=4)
    key2index = w2v.wv.key_to_index
    vectors = w2v.wv.vectors
    channels = []
    for label, code in labels_code.items():
        emb_node2vec = vectors[key2index[label]]
        line_vec = sentence_embedding(code)
        line_vec = np.array(line_vec)

        channels.append(line_vec + emb_node2vec)
    return (channels, channels, channels)


if __name__ == '__main__':
    image_generation("/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/cfgs/pdgs/1000_1.dot")

