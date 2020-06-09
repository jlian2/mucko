import os
import json
import yaml
import argparse
import numpy as np

from math import log
import dgl
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bisect import bisect
from util.vocabulary import Vocabulary
from util.checkpointing import CheckpointManager, load_checkpoint
from model_fvqa.model import CMGCNnet
from model_fvqa.train_dataset import FvqaTrainDataset
from model_fvqa.test_dataset import FvqaTestDataset


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train():
   
    parser = argparse.ArgumentParser()
    # 配置文件
    parser.add_argument(
        "--config-yml",
        default="exp_fvqa/exp2.yml",
        help=
        "Path to a config file listing reader, model and solver parameters.")

    parser.add_argument("--cpu-workers",
                        type=int,
                        default=8,
                        help="Number of CPU workers for dataloader.")

    parser.add_argument(
        "--save-dirpath",
        default="fvqa/exp_data/checkpoints",
        help=
        "Path of directory to create checkpoint directory and save checkpoints."
    )

    parser.add_argument(
        "--load-pthpath",
        default="",
        help="To continue training, path to .pth file of saved checkpoint.")

    parser.add_argument("--gpus", default="", help="gpus")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Whether to validate on val split after every epoch.")

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Whether to validate on val split after every epoch.")

    args = parser.parse_args()

    # set mannual seed
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    cudnn.benchmark = True
    cudnn.deterministic = True

    config = yaml.load(open(args.config_yml))

    device = torch.device("cuda:0") if args.gpus != "cpu" else torch.device(
        "cpu")

    # Print config and args.
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))

 
    print('Loading TrainDataset...')
    train_dataset = FvqaTrainDataset(config, overfit=args.overfit)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['solver']['batch_size'],
                                  num_workers=args.cpu_workers,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    if args.validate:
        print('Loading TestDataset...')
        val_dataset = FvqaTestDataset(config, overfit=args.overfit)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['solver']['batch_size'],
                                    num_workers=args.cpu_workers,
                                    shuffle=True,
                                    collate_fn=collate_fn)


    print('Loading glove...')
    que_vocab = Vocabulary(config['dataset']['word2id_path'])
    glove = np.load(config['dataset']['glove_vec_path'])
    glove = torch.Tensor(glove)


    print('Building Model...')
    model = CMGCNnet(config,
                     que_vocabulary=que_vocab,
                     glove=glove,
                     device=device)

    if torch.cuda.device_count() > 1 and args.gpus != "cpu":
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(model)


    iterations = len(train_dataset) // config["solver"]["batch_size"] + 1

    def lr_lambda_fun(current_iteration: int) -> float:
   
        current_epoch = float(current_iteration) / iterations
        if current_epoch <= config["solver"]["warmup_epochs"]:
            alpha = current_epoch / float(config["solver"]["warmup_epochs"])
            return config["solver"]["warmup_factor"] * (1. - alpha) + alpha
        else:
            idx = bisect(config["solver"]["lr_milestones"], current_epoch)
            return pow(config["solver"]["lr_gamma"], idx)


    optimizer = optim.Adamax(model.parameters(),
                             lr=config["solver"]["initial_lr"])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
    T = iterations * (config["solver"]["num_epochs"] -
                      config["solver"]["warmup_epochs"] + 1)
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, int(T), eta_min=config["solver"]["eta_min"], last_epoch=-1)

   
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)
    checkpoint_manager = CheckpointManager(model,
                                           optimizer,
                                           args.save_dirpath,
                                           config=config)


    if args.load_pthpath == "":
        start_epoch = 0
    else:

        start_epoch = int(args.load_pthpath.split("_")[-1][:-4])

        model_state_dict, optimizer_state_dict = load_checkpoint(
            args.load_pthpath)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        print("Loading resume model from {}...".format(args.load_pthpath))


    global_iteration_step = start_epoch * iterations

    for epoch in range(start_epoch, config['solver']['num_epochs']):

        print(f"\nTraining for epoch {epoch}:")

        train_answers = []
        train_preds = []

        for i, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            fact_batch_graph = model(batch)
            batch_loss = cal_batch_loss(fact_batch_graph,
                                        batch,
                                        device,
                                        neg_weight=0.1,
                                        pos_weight=0.9)

            batch_loss.backward()
            optimizer.step()

            fact_graphs = dgl.unbatch(fact_batch_graph)
            for i, fact_graph in enumerate(fact_graphs):
                train_pred = fact_graph.ndata['h'].squeeze()  # (num_nodes,1)
                train_preds.append(train_pred)  # [(num_nodes,)]
                train_answers.append(batch['facts_answer_id_list'][i])

            summary_writer.add_scalar('train/loss', batch_loss,
                                      global_iteration_step)
            summary_writer.add_scalar("train/lr",
                                      optimizer.param_groups[0]["lr"],
                                      global_iteration_step)
            summary_writer.add_text('train/loss', str(batch_loss.item()),
                                    global_iteration_step)
            summary_writer.add_text('train/lr',
                                    str(optimizer.param_groups[0]["lr"]),
                                    global_iteration_step)

            if global_iteration_step <= iterations * config["solver"][
                "warmup_epochs"]:
                scheduler.step(global_iteration_step)
            else:
                global_iteration_step_in_2 = iterations * config["solver"][
                    "warmup_epochs"] + 1 - global_iteration_step
                scheduler2.step(int(global_iteration_step_in_2))

            global_iteration_step = global_iteration_step + 1
            torch.cuda.empty_cache()


        checkpoint_manager.step()
        train_acc_1, train_acc_3 = cal_acc(
            train_answers, train_preds)
        print(
            "trainacc@1={:.2%} & trainacc@3={:.2%} "
                .format(train_acc_1, train_acc_3))
        summary_writer.add_scalars(
            'train/acc', {
                'acc@1': train_acc_1,
                'acc@3': train_acc_3

            }, epoch)


        if args.validate:
            model.eval()
            answers = []  # [batch_answers,...]
            preds = []  # [batch_preds,...]
            print(f"\nValidation after epoch {epoch}:")
            for i, batch in enumerate(tqdm(val_dataloader)):
                with torch.no_grad():
                    fact_batch_graph = model(batch)
                batch_loss = cal_batch_loss(fact_batch_graph,
                                            batch,
                                            device,
                                            neg_weight=0.1,
                                            pos_weight=0.9)

                summary_writer.add_scalar('test/loss', batch_loss, epoch)
                fact_graphs = dgl.unbatch(fact_batch_graph)
                for i, fact_graph in enumerate(fact_graphs):
                    pred = fact_graph.ndata['h'].squeeze()  # (num_nodes,1)
                    preds.append(pred)  # [(num_nodes,)]
                    answers.append(batch['facts_answer_id_list'][i])

            acc_1, acc_3 = cal_acc(answers, preds)
            print("acc@1={:.2%} & acc@3={:.2%} ".
                  format(acc_1, acc_3))
            summary_writer.add_scalars('test/acc', {
                'acc@1': acc_1,
                'acc@3': acc_3
            }, epoch)

            model.train()
            torch.cuda.empty_cache()
    print('Train finished !!!')
    summary_writer.close()


def cal_batch_loss(fact_batch_graph, batch, device, pos_weight, neg_weight):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).to(device)

    for i, fact_graph in enumerate(fact_graphs):
        class_weight = torch.FloatTensor([neg_weight, pos_weight])
        pred = fact_graph.ndata['h'].view(1, -1)  # (n,1)
        answer = torch.FloatTensor(answers[i]).view(1, -1).to(device)
        pred = pred.squeeze()
        answer = answer.squeeze()
        weight = class_weight[answer.long()].to(device)
        loss_fn = torch.nn.BCELoss(weight=weight)
        loss = loss_fn(pred, answer)
        batch_loss = batch_loss + loss

    return batch_loss / len(answers)


def focal_loss(fact_batch_graph, batch, device, alpha=0.5, gamma=2):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).float().to(device)

    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h'].squeeze()
        target = torch.FloatTensor(answers[i]).to(device).squeeze()
        loss = -1 * alpha * ((1 - pred) ** gamma) * target * torch.log(pred) - (1 - alpha) * (target ** gamma) * (
                    1 - pred) * torch.log(1 - pred)
        batch_loss = batch_loss+loss.mean()
    return batch_loss/len(answers)


def cal_acc(answers, preds):
    all_num = len(preds)
    acc_num_1 = 0
    acc_num_3 = 0
 

    for i, answer_id in enumerate(answers):
        pred = preds[i]  # (num_nodes)
        # top@1
        _, idx_1 = torch.topk(pred, k=1)
        if idx_1.item() == answer_id:
            acc_num_1 = acc_num_1 + 1

        # top@3
        _, idx_3 = torch.topk(pred, k=3)
        if answer_id in idx_3:
            acc_num_3 = acc_num_3 + 1

    return acc_num_1 / all_num, acc_num_3 / all_num


def collate_fn(batch):
    res = {}
    id_list = []
    question_list = []

    question_length_list = []
    features_list = []
    img_relations_list = []

    num_nodes_list = []
    # facts_nodes_list = []
    facts_features_list = []
    facts_e1ids_list = []
    facts_e2ids_list = []
    facts_answer_list = []
    facts_answer_id_list = []

    semantic_n_features_list = []
    semantic_e1ids_list = []
    semantic_e2ids_list = []
    semantic_e_features_list = []
    semantic_num_nodes_list = []



    for item in batch:
        # question
        id = item['id']  # scalar
        id_list.append(id)
        question = item['question']  # (max_len,)
        question_list.append(question)
      
        question_length = item['question_length']  # scalar
        question_length_list.append(question_length)

        features = item['features']  # (36,2048)
        features_list.append(features)
      
        img_relations = item['img_relations']
        img_relations_list.append(img_relations)

        num_nodes = item['facts_num_nodes']  # scalar
        num_nodes_list.append(num_nodes)
     
        facts_features = item['facts_features']  # (num,1024)
        facts_features_list.append(facts_features)
        facts_e1ids = item['facts_e1ids']  # (num_nodes,)
        facts_e1ids_list.append(facts_e1ids)
        facts_e2ids = item['facts_e2ids']  # (num_nodes,)
        facts_e2ids_list.append(facts_e2ids)
        # (num_nodes,)  one-hot
        facts_answer = item['facts_answer']
        facts_answer_list.append(facts_answer)
        facts_answer_id = item['facts_answer_id']  # scalar
        facts_answer_id_list.append(facts_answer_id)

        semantic_num_nodes = item['semantic_num_nodes']
        semantic_num_nodes_list.append(semantic_num_nodes)
        semantic_n_features = item['semantic_n_features']
        semantic_n_features_list.append(semantic_n_features)
        semantic_e1ids = item['semantic_e1ids']
        semantic_e1ids_list.append(semantic_e1ids)
        semantic_e2ids = item['semantic_e2ids']
        semantic_e2ids_list.append(semantic_e2ids)
        semantic_e_features = item['semantic_e_features']
        semantic_e_features_list.append(semantic_e_features)

    res['id_list'] = id_list

    res['question_list'] = question_list
  
    res['question_length_list'] = question_length_list
    res['features_list'] = features_list

    res['img_relations_list'] = img_relations_list

    res['facts_num_nodes_list'] = num_nodes_list

    res['facts_features_list'] = facts_features_list
    res['facts_e1ids_list'] = facts_e1ids_list
    res['facts_e2ids_list'] = facts_e2ids_list
    res['facts_answer_list'] = facts_answer_list
    res['facts_answer_id_list'] = facts_answer_id_list

    res['semantic_n_features_list'] = semantic_n_features_list
    res['semantic_e1ids_list'] = semantic_e1ids_list
    res['semantic_e2ids_list'] = semantic_e2ids_list
    res['semantic_e_features_list'] = semantic_e_features_list
    res['semantic_num_nodes_list'] = semantic_num_nodes_list

    return res


def pad_sequences(self, sequence):

    sequence = sequence[:self.config['dataset']['max_sequence_lengtn']]

    padding = np.zeros(self.config['dataset']['max_sequence_lengtn'])
    padding[:len(sequence)] = np.array(sequence)
    return torch.tensor(padding)


if __name__ == "__main__":
    train()
