from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import json
import pickle

if __name__ == '__main__':

    glove_300d_file = ''
    glove_300d_word2vec_file = ''

    print('loading glove model...')
    glove_model = KeyedVectors.load_word2vec_format(glove_300d_word2vec_file, binary=False)
    print('finished load glove model...')

    with open('semantic_raw.json', 'r') as f:
        raws = json.load(f)

    res={}

    for sem_dict in raws:
        nodes = []
        edge_set = set()
        e2id = {}
        id2e = {}
        e1ids = []
        e2ids = []
        e_features = []
        n_features = []

        image_id = sem_dict['image_id']
        ref_tuples = sem_dict['ref_tuples']
        for tuple_dict in ref_tuples:
            tuple = tuple_dict['tuple']
            if len(tuple)==1:
                continue
            if len(tuple) == 3:
                e1 = tuple[0]
                r = tuple[1]
                e2 = tuple[2]
            if len(tuple) == 2:
                e1 = tuple[1]
                r = 'property'
                e2 = tuple[0]
            try:
                edge_embed = glove_model[r]
            except KeyError:
                edge_embed=np.zeros(300,dtype=np.float32)
            e1s=e1.split('/')
            e2s=e2.split('/')
            for e1 in e1s:
                for e2 in e2s:
                    if e1 not in nodes:
                        nodes.append(e1)
                    if e2 not in nodes:
                        nodes.append(e2)

                    if (e1,e2) not in edge_set:
                        edge_set.add((e1, e2))

                        e_features.append(edge_embed)

        for i, e in enumerate(nodes):
            e2id[e] = i
            id2e[i] = e
            try:
                node_embed = glove_model[e]
            except KeyError:
                node_embed=np.zeros(300,dtype=np.float32)
            n_features.append(node_embed)

        # 将 e 转成 id 构造边数组
        for e in edge_set:
            e1ids.append(e2id[e[0]])
            e2ids.append(e2id[e[1]])

        print(image_id)
        print(nodes)
        item = {}
        item['nodes'] = nodes
        item['n_features'] = np.stack(n_features)
        item['e2id'] = e2id
        item['id2e'] = id2e
        item['e1ids'] = e1ids
        item['e2ids'] = e2ids
        item['e_features'] = np.stack(e_features)

        res[image_id]=item

    with open('all_semantic_graph.pickle','wb') as f:
        pickle.dump(res,f)

    print('finished!')





    with open('all_semantic_graph.pickle', 'rb') as f:
        raw = pickle.load(f, encoding='iso-8859-1')

    with open('all_qs_rel_dict_release.json', 'r') as f:
        qa_image_rel_dicts = json.load(f)

    train_images = []
    test_images = []

    all_qid_img = {}

    for qid, items in qa_image_rel_dicts.items():
        image = items['img_file']
        train_images.append(image)
        # all_qid_img[]

    with open('train_image_ids.json', 'w') as f:
        json.dump(train_images, f)

    train_sem_graphs = []
    test_sem_graphs = []
    for image_id in train_images:
        train_sem_graphs.append(raw[image_id])

    with open('test_qids.json', 'r') as f:
        test_qids = json.load(f)

    for qid in test_qids:
        image_id = qa_image_rel_dicts[qid]['img_file']
        test_sem_graphs.append(raw[image_id])

    with open('train_semantic_graph.pickle', 'wb') as f:
        pickle.dump(train_sem_graphs, f)
    print('finished!')

    with open('test_semantic_graph.pickle', 'wb') as f:
        pickle.dump(test_sem_graphs, f)
        print('finished!')

