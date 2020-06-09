import json
from tqdm import tqdm
import re
import yaml
import pickle
import numpy as np
import argparse

_special_chars = re.compile('[^a-z0-9 ]*')

config = yaml.load(open("/home/yujing/zzh/fvqa2/exp_fvqa/exp2.yml"))
config = config['dataset']

test_split_file = "fvqa/exp_data/data/Name_Lists/test_list_0.txt"


# 将qid-img多对一的关系转换成img-qid一对多的关系
def extract_image_id():
    all_facts_file = "fvqa/exp_data/data/all_fact_triples_release.json"
    all_qa_path = "fvqa/exp_data/data/all_qs_dict_release.json"
    all_qa_rel_path = "fvqa/exp_data/data/all_qs_rel_dict_release.json"
    all_image_qid_path = "fvqa/exp_data/data/all_image_qid.json"
    with open(all_qa_path, 'r') as f:
        qa_image_dicts = json.load(f)
    with open(all_facts_file, 'r') as fd:
        facts = json.load(fd)

    image_qa_dict = {}

    for q_id, item in tqdm(qa_image_dicts.items()):
        image_id = item['img_file']
        qids = []
        if not image_id in image_qa_dict:
            image_qa_dict[image_id] = {}

        if 'qids' in image_qa_dict[image_id]:
            image_qa_dict[image_id]['qids'].append(q_id)
        else:
            image_qa_dict[image_id]['qids'] = [q_id]

    with open(all_image_qid_path, 'w') as f:
        json.dump(image_qa_dict, f)


# 去掉caption中的<UNK>
def handel_caption_file():
    caption_dict = {}
    with open(config['image_captions_path'], 'r') as f:
        captions = json.load(f)
        for i in range(len(captions)):
            refs = []
            caption_dict[captions[i]['image_id']] = {}
            for ref in captions[i]['refs']:
                if not 'UNK' in ref:
                    refs.append(ref)
            captions[i]['refs'] = refs
            caption_dict[captions[i]['image_id']]['refs'] = refs
    with open("fvqa/exp_data/data/captions.json",
              'w') as f:
        json.dump(caption_dict, f)
    with open(
            "fvqa/exp_data/data/captions_for_graph.json",
            'w') as f:
        json.dump(captions, f)


# 根据数据集提供的test image list 来提取出需要的数据
def extract_test():
    test_split_file = "fvqa/exp_data/data/Name_Lists/test_list_1.txt"

    # 所有的问题集合
    with open(config['all_qa_rel_path'], 'r') as f:
        qa_image_rel_dicts = json.load(f)

    # 所有的facts集合
    with open(config['all_facts_path'], 'r') as fd:
        facts = json.load(fd)

    # test数据
    with open(test_split_file, 'r') as f:
        test_split_images = f.read().splitlines()

    # 所有的image对应的qid
    with open(config['all_image_qid_path'], 'r') as f:
        image_qa_dict = json.load(f)

    # 所有的image对应的caption
    with open(config['image_captions_path'], 'r') as f:
        captions = json.load(f)

    # 所有的image对应的features
    with open(config['image_features_path'], 'rb') as f:
        features = pickle.load(f, encoding='iso-8859-1')

    # 所有的image对应的sem graph
    with open("fvqa/exp_data/data/all_semantic_graph.pickle", 'rb') as f:
        all_sem_graphs = pickle.load(f, encoding='iso-8859-1')

    test_qids = []
    test_questions = []
    test_answers = []
    test_relations = []
    test_gt_facts = []
    test_facts = []
    test_captions = []
    test_features = []
    test_labels = []
    test_bboxes = []
    test_whs = []
    test_images=[]
    test_sem_graphs=[]

    for image in tqdm(test_split_images):
        qids = image_qa_dict[image]['qids']
        test_qids = test_qids + qids
        # image对应的caption
        caption = captions[image]['refs']

        # image对应的 sem graph
        sem_graph = all_sem_graphs[image]

        # image对应的features
        feature = features[image]['features']

        w = features[image]['image_w']
        h = features[image]['image_h']

        # image对应的labels
        labels = features[image]['labels']
        labels = list(set(labels))

        # image对应的bboxes
        boxes = features[image]['boxes']

        for qid in qids:
            #qid所对应的问题
            question = qa_image_rel_dicts[qid]['question']
            question = question.strip().lower().replace('?', '')
            question = _special_chars.sub('', question).strip()

            #  qid 所对应的回答
            answer = qa_image_rel_dicts[qid]['answer']
            answer = answer.lower().replace('a ', '').strip()
            answer = _special_chars.sub('', answer).strip()

            # qid 所对应的relation
            gt_relation = qa_image_rel_dicts[qid]['relation']
            test_relations.append(gt_relation)

            # q_id 对应的gt fact
            gt_fact_id = qa_image_rel_dicts[qid]['fact'][0]
            gt_e1 = facts[gt_fact_id]['e1_label']
            gt_e2 = facts[gt_fact_id]['e2_label']
            gt_surface = facts[gt_fact_id]["surface"].lower()


            test_gt_facts.append([gt_e1, gt_e2, gt_relation, gt_surface])
            test_questions.append(question)
            test_answers.append(answer)
            test_captions.append(caption)
            test_features.append(feature)
            test_labels.append(labels)
            test_bboxes.append(boxes)
            test_whs.append([w, h])
            test_images.append(image)
            test_sem_graphs.append(sem_graph)


    with open(
            'fvqa/exp_data/data/test1/test_qids.json',
            'w') as f:
        json.dump(test_qids, f)
        print(len(test_qids))
    with open(
            'fvqa/exp_data/data/test1/test_questions.json',
            'w') as f:
        json.dump(test_questions, f)
        print(len(test_questions))
    with open(
            'fvqa/exp_data/data/test1/test_answers.json',
            'w') as f:
        json.dump(test_answers, f)
        print(len(test_answers))
    with open(
            'fvqa/exp_data/data/test1/test_captions.json',
            'w') as f:
        json.dump(test_captions, f)
        print(len(test_captions))
    with open(
            'fvqa/exp_data/data/test1/test_whs.json',
            'w') as f:
        json.dump(test_whs, f)
        print(len(test_whs))

    np.save(
        'fvqa/exp_data/data/test1/test_features.npy',
        np.array(test_features))
    print(len(test_features))

    with open(
            'fvqa/exp_data/data/test1/test_labels.json',
            'w') as f:
        json.dump(test_labels, f)
        print(len(test_labels))

    np.save(
        'fvqa/exp_data/data/test1/test_bboxes.npy',
        np.array(test_bboxes))
    print(len(test_bboxes))

    with open(
            'fvqa/exp_data/data/test1/test_gt_facts.json',
            'w') as f:
        json.dump(test_gt_facts, f)
        print(len(test_gt_facts))
    with open(
            'fvqa/exp_data/data/test1/test_images.json',
            'w') as f:
        json.dump(test_gt_facts, f)
        print(len(test_gt_facts))

    with open("fvqa/exp_data/data/test1/test_semantic_graph.pickle", 'wb') as f:
        pickle.dump(test_sem_graphs, f)
        print('finish')


# 根据数据集提供的train image list 来提取出需要的数据
def extract_train():
    # 所有的问题集合
    with open(config['all_qa_rel_path'], 'r') as f:
        qa_image_rel_dicts = json.load(f)

    # 所有的facts集合
    with open(config['all_facts_path'], 'r') as fd:
        facts = json.load(fd)

    # test数据
    with open(test_split_file, 'r') as f:
        test_split_images = f.read().splitlines()

    # 所有的image对应的qid
    with open(config['all_image_qid_path'], 'r') as f:
        image_qa_dict = json.load(f)

    # 所有的image对应的caption
    with open(config['image_captions_path'], 'r') as f:
        captions = json.load(f)

    # 所有的image对应的features
    with open(config['image_features_path'], 'rb') as f:
        features = pickle.load(f, encoding='iso-8859-1')

    train_qids = []
    train_questions = []
    train_answers = []
    train_relations = []
    train_labels = []
    train_captions = []
    train_bboxes = []
    train_features = []
    train_gt_facts = []
    train_facts = []
    train_facts_id = []
    train_whs = []

    for qid, items in tqdm(qa_image_rel_dicts.items()):
        image = items['img_file']

        # question = items['question']
        # question = question.strip().lower().replace('?', '')
        # question = _special_chars.sub('', question).strip()

        # answer = items['answer']
        # answer = answer.lower().replace('a ', '').strip()
        # answer = _special_chars.sub('', answer).strip()

        gt_relation = items['relation']

        # # image对应的caption
        # caption = captions[image]['refs']

        # # image对应的features
        # feature = features[image]['features']

        # w = features[image]['image_w']
        # h = features[image]['image_h']

        # image对应的labels
        labels = features[image]['labels']
        labels = list(set(labels))

        # image对应的bboxes
        # boxes = features[image]['boxes']

        # q_id 对应的gt fact
        gt_fact_id = items['fact'][0]
        gt_e1 = facts[gt_fact_id]['e1_label']
        gt_e2 = facts[gt_fact_id]['e2_label']
        gt_surface = facts[gt_fact_id]["surface"].lower()

        # q_id 抽取出的facts
        extract_facts = []
        extract_facts_id = []
        extract_facts.append([gt_e1, gt_e2, gt_relation, gt_surface])
        for label in labels:
            for fact_id, fact_item in facts.items():
                e1 = fact_item["e1_label"].lower()
                e2 = fact_item["e2_label"].lower()
                r = fact_item["r"]
                surface = fact_item["surface"].lower()
                if (label == e1 or label == e2
                    ) and gt_relation == r and e1 != '' and e2 != '':
                    extract_facts.append([e1, e2, r, surface])
                    extract_facts_id.append(fact_id)

        train_facts.append(extract_facts)
        train_gt_facts.append([gt_e1, gt_e2, gt_relation, gt_surface])
        train_facts_id.append(extract_facts_id)

        # train_labels.append(labels)
        # train_bboxes.append(boxes)
        # train_features.append(feature)
        # train_relations.append(gt_relation)
        # train_qids.append(qid)
        # train_answers.append(answer)
        # train_questions.append(question)
        # train_captions.append(caption)
        # train_whs.append([w, h])

    # with open(
    #         'fvqa/exp_data/data/train0/train_qids.json',
    #         'w') as f:
    #     json.dump(train_qids, f)
    #     print(len(train_qids))

    # with open(
    #         'fvqa/exp_data/data/train0/train_questions.json',
    #         'w') as f:
    #     json.dump(train_questions, f)
    #     print(len(train_questions))
    # with open(
    #         'fvqa/exp_data/data/train0/train_answers.json',
    #         'w') as f:
    #     json.dump(train_answers, f)
    #     print(len(train_answers))
    # with open(
    #         'fvqa/exp_data/data/train0/train_captions.json',
    #         'w') as f:
    #     json.dump(train_captions, f)
    #     print(len(train_captions))

    # np.save(
    #     'fvqa/exp_data/data/train0/train_features.npy',
    #     np.array(train_features))
    # print(len(train_features))

    # with open(
    #         'fvqa/exp_data/data/train0/train_labels.json',
    #         'w') as f:
    #     json.dump(train_labels, f)
    #     print(len(train_labels))

    # np.save(
    #     'fvqa/exp_data/data/train0/train_bboxes.npy',
    #     np.array(train_bboxes))
    # print(len(train_bboxes))
    # with open(
    #         'fvqa/exp_data/data/train0/train_whs.json',
    #         'w') as f:
    #     json.dump(train_whs, f)
    #     print(len(train_whs))

    with open(
            'fvqa/exp_data/data/train0/train_gt_facts.json',
            'w') as f:
        json.dump(train_gt_facts, f)
        print(len(train_gt_facts))
    with open(
            'fvqa/exp_data/data/train0/train_facts.json',
            'w') as f:
        json.dump(train_facts, f)
        print(len(train_facts))
    with open(
            'fvqa/exp_data/data/train0/train_facts_id.json',
            'w') as f:
        json.dump(train_facts_id, f)
        print(len(train_facts_id))


if __name__ == '__main__':

    extract_test()
