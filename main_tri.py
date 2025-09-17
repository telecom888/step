import os
import os.path as osp
import shutil
import time
import csv
import math
import pickle
import numpy as np

import torch
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup

# 请确保你的 StepwisePivotTransformer 类已在此文件中定义，或者可以被正确导入。
# 这里假设它和旧模型在同一个 models.py 文件中。
# from models import MLPModel, GCNModel, AttentionModel, TransformerRMLPModel, StepwisePivotTransformer
from decode import StepwisePivotTransformer

from dataloaders import get_dataloader, GenerateData, get_graph_data, get_attention_graph_data, GenerateDataAttention, get_attention_dataloader

import argparse
import multiprocessing

import torch.nn.functional as F


# 定义双向三元组损失函数
def bidirectional_triplet_loss(text_out, chem_out, f_out, margin=0.3, lambda_param=2e-3):
    batch_size = text_out.size(0)

    # 计算文本和化学分子嵌入之间的成对距离
    dist_matrix = F.pairwise_distance(text_out.unsqueeze(1).expand(-1, batch_size, -1),
                                      chem_out.unsqueeze(0).expand(batch_size, -1, -1), p=2)

    # 正样本距离（对角线元素）
    pos_dist = torch.diag(dist_matrix)

    # 负样本距离（非对角线元素）
    neg_dist_text = dist_matrix.clone()
    neg_dist_chem = dist_matrix.clone()
    torch.diagonal(neg_dist_text).fill_(float('inf'))
    torch.diagonal(neg_dist_chem).fill_(float('inf'))
    neg_dist_text_min, _ = neg_dist_text.min(dim=1)
    neg_dist_chem_min, _ = neg_dist_chem.min(dim=0)

    # 文本到化学分子的三元组损失
    loss_text = F.relu(pos_dist - neg_dist_text_min + margin)
    # 化学分子到文本的三元组损失
    loss_chem = F.relu(pos_dist - neg_dist_chem_min + margin)

    # 双向三元组损失
    loss = (loss_text.mean() + loss_chem.mean()) / 2

    # 正则化项
    reg_loss = lambda_param * (torch.norm(text_out, p=2) + torch.norm(chem_out, p=2))

    total_loss = loss + reg_loss

    return total_loss


# 定义一个新的主函数来替换原来的 main()
def new_main():
    parser = argparse.ArgumentParser(description='Run Text2Mol with Stepwise Pivot Transformer')
    parser.add_argument('--data', metavar='data', type=str, default='./data',
                        help='directory where data is located')
    parser.add_argument('--output_path', metavar='output_path', type=str, default='./output',
                        help='directory where data is located')
    parser.add_argument('--model', type=str, default='StepwisePivotTransformer', nargs='?',
                        help="model type from 'MLP', 'GCN', 'Attention', 'StepwisePivotTransformer'")
    parser.add_argument('--mol_trunc_length', type=int, nargs='?', default=512,
                        help='Molecule truncation length.')
    parser.add_argument('--text_trunc_length', type=int, nargs='?', default=256,
                        help='Text truncation length.')
    parser.add_argument('--num_warmup_steps', type=int, nargs='?', default=1000,
                        help='Number of warmup steps.')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs to train model.')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Size of data batch.')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-4,
                        help='learning rate')
    parser.add_argument('--bert_lr', type=float, nargs='?', default=3e-5,
                        help='Size of data batch.')

    args = parser.parse_args()
    data_path = args.data
    output_path = args.output_path
    MODEL = args.model

    BATCH_SIZE = args.batch_size
    epochs = args.epochs

    init_lr = args.lr
    bert_lr = args.bert_lr
    num_warmup_steps = args.num_warmup_steps
    text_trunc_length = args.text_trunc_length
    mol_trunc_length = args.mol_trunc_length

    # 检查命令行参数，如果未提供则使用默认值
    if data_path is None:
        print("Warning: '--data' argument not provided. Using default path './data'.")
        data_path = './data'
    if output_path is None:
        print("Warning: '--output_path' argument not provided. Using default path './output'.")
        output_path = './output'

    path_token_embs = osp.join(data_path, "token_embedding_dict.npy")
    path_train = osp.join(data_path, "training.txt")
    path_val = osp.join(data_path, "val.txt")
    path_test = osp.join(data_path, "test.txt")
    path_molecules = osp.join(data_path, "ChEBI_defintions_substructure_corpus.cp")
    graph_data_path = osp.join(data_path, "mol_graphs.zip")

    # 根据新的模型名称来实例化模型
    if MODEL == "StepwisePivotTransformer":
        gd = GenerateData(text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs)

        # Parameters
        params = {'batch_size': BATCH_SIZE,
                  'num_workers': 0}

        training_generator, validation_generator, test_generator = get_dataloader(gd, params)

        graph_batcher_tr, graph_batcher_val, graph_batcher_test = get_graph_data(gd, graph_data_path)

        # 实例化 StepwisePivotTransformer，并添加 num_layers 参数
        model = StepwisePivotTransformer(
            num_node_features=graph_batcher_tr.dataset.num_node_features,
            ninp=768,
            nout=300,
            nhid=600,
            graph_hidden_channels=600,
            heads=4,
            num_layers=3
        )
    else:
        raise ValueError(f"Unsupported model type: {MODEL}. Please use 'StepwisePivotTransformer'.")

    # 获取BERT参数
    bert_params = list(model.text_transformer_model.parameters())

    # 获取模型全部参数，然后从中筛选出非BERT参数
    all_params = list(model.parameters())
    other_params = [p for p in all_params if id(p) not in [id(bp) for bp in bert_params]]

    optimizer = optim.Adam([
        {'params': other_params},  # 使用新的 'other_params' 列表
        {'params': bert_params, 'lr': bert_lr}
    ], lr=init_lr)

    num_training_steps = epochs * len(training_generator) - num_warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 移除 set_device 方法，使用标准的 .to(device)
    model = model.to(device)

    train_losses = []
    val_losses = []

    train_acc = []
    val_acc = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 定义损失函数的参数
    margin = 0.3
    lambda_param = 2e-3

    # Loop over epochs
    for epoch in range(epochs):
        # Training
        start_time = time.time()
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, d in enumerate(training_generator):
            batch, labels = d
            # Transfer to GPU
            text_mask = batch['text']['attention_mask'].bool()
            text = batch['text']['input_ids'].to(device)
            text_mask = text_mask.to(device)
            # 新模型不再需要 mol2vec 输入

            if MODEL == "StepwisePivotTransformer":
                graph_batch = graph_batcher_tr(d[0]['molecule']['cid']).to(device)
                # 修改模型调用，接收三个返回值
                f_c, f_text, f_graph = model(text, graph_batch, text_mask)

                # 使用领域特定特征 f_text 和 f_graph 来计算损失
                loss = bidirectional_triplet_loss(f_text, f_graph, margin=margin, lambda_param=lambda_param)
                running_loss += loss.item()
                print("sample ", i+1 ," loss ", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % 100 == 0:
                print(i + 1, "batches trained. Avg loss:\t", running_loss / (i + 1), ". Avg ms/step =",
                      1000 * (time.time() - start_time) / (i + 1))

        train_losses.append(running_loss / (i + 1))
        train_acc.append(running_acc / (i + 1))

        print("Epoch", epoch + 1, "training loss:\t\t", train_losses[-1], ". Time =", (time.time() - start_time),
              "seconds.")

        # Validation
        model.eval()
        with torch.set_grad_enabled(False):
            start_time = time.time()
            running_loss = 0.0
            running_acc = 0.0
            for i, d in enumerate(validation_generator):
                batch, labels = d
                # Transfer to GPU
                text_mask = batch['text']['attention_mask'].bool()
                text = batch['text']['input_ids'].to(device)
                text_mask = text_mask.to(device)

                if MODEL == "StepwisePivotTransformer":
                    graph_batch = graph_batcher_val(d[0]['molecule']['cid']).to(device)
                    # 修改模型调用，接收三个返回值
                    f_c, f_text, f_graph = model(text, graph_batch, text_mask)
                    # 使用领域特定特征 f_text 和 f_graph 来计算损失
                    loss = bidirectional_triplet_loss(f_text, f_graph, margin=margin, lambda_param=lambda_param)
                    running_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print(i + 1, "batches eval. Avg loss:\t", running_loss / (i + 1), ". Avg ms/step =",
                          1000 * (time.time() - start_time) / (i + 1))

            val_losses.append(running_loss / (i + 1))
            val_acc.append(running_acc / (i + 1))

            min_loss = np.min(val_losses)
            if val_losses[-1] == min_loss:
                torch.save(model.state_dict(),
                           output_path + 'weights_pretrained.{epoch:02d}-{min_loss:.2f}.pt'.format(epoch=epoch + 1,
                                                                                                   min_loss=min_loss))

        print("Epoch", epoch + 1, "validation loss:\t", val_losses[-1], ". Time =", (time.time() - start_time),
              "seconds.")

    torch.save(model.state_dict(), output_path + "final_weights." + str(epochs) + ".pt")

    cids_train = np.array([])
    cids_val = np.array([])
    cids_test = np.array([])
    chem_embeddings_train = np.array([])
    text_embeddings_train = np.array([])
    chem_embeddings_val = np.array([])
    text_embeddings_val = np.array([])
    chem_embeddings_test = np.array([])
    text_embeddings_test = np.array([])

    if MODEL != "Attention":  # Store embeddings:
        def get_emb(d, graph_batcher=None):
            with torch.no_grad():
                cid = np.array([d['cid']])
                text_mask = torch.Tensor(d['input']['text']['attention_mask']).bool().reshape(1, -1).to(device)
                text = torch.Tensor(d['input']['text']['input_ids']).long().reshape(1, -1).to(device)

                if MODEL == "StepwisePivotTransformer":
                    graph_batch = graph_batcher([d['input']['molecule']['cid']]).to(device)
                    graph_batch.edge_index = graph_batch.edge_index.reshape((2, -1))
                    # 修改模型调用，接收三个返回值
                    f_c, f_text, f_graph = model(text, graph_batch, text_mask)
                    # 将 f_text 和 f_graph 赋值给 text_emb 和 chem_emb
                    text_emb = f_text
                    chem_emb = f_graph
                else:
                    # 此处为旧模型逻辑，为防止报错暂时跳过
                    return None, None, None

                chem_emb = chem_emb.cpu().numpy()
                text_emb = text_emb.cpu().numpy()

            return cid, chem_emb, text_emb

        for i, d in enumerate(gd.generate_examples_train()):
            if MODEL == "StepwisePivotTransformer":
                cid, chem_emb, text_emb = get_emb(d, graph_batcher_tr)
            else:
                continue

            # 仅在获取到有效嵌入时才进行拼接
            if cid is not None:
                cids_train = np.concatenate((cids_train, cid)) if cids_train.size else cid
                chem_embeddings_train = np.concatenate(
                    (chem_embeddings_train, chem_emb)) if chem_embeddings_train.size else chem_emb
                text_embeddings_train = np.concatenate(
                    (text_embeddings_train, text_emb)) if text_embeddings_train.size else text_emb

            if (i + 1) % 1000 == 0:
                print(i + 1, "embeddings processed")

        print("Training Embeddings done:", cids_train.shape, chem_embeddings_train.shape)

        for d in gd.generate_examples_val():
            if MODEL == "StepwisePivotTransformer":
                cid, chem_emb, text_emb = get_emb(d, graph_batcher_val)
            else:
                continue

            if cid is not None:
                cids_val = np.concatenate((cids_val, cid)) if cids_val.size else cid
                chem_embeddings_val = np.concatenate(
                    (chem_embeddings_val, chem_emb)) if chem_embeddings_val.size else chem_emb
                text_embeddings_val = np.concatenate(
                    (text_embeddings_val, text_emb)) if text_embeddings_val.size else text_emb

        print("Validation Embeddings done:", cids_val.shape, chem_embeddings_val.shape)

        for d in gd.generate_examples_test():
            if MODEL == "StepwisePivotTransformer":
                cid, chem_emb, text_emb = get_emb(d, graph_batcher_test)
            else:
                continue

            if cid is not None:
                cids_test = np.concatenate((cids_test, cid)) if cids_test.size else cid
                chem_embeddings_test = np.concatenate(
                    (chem_embeddings_test, chem_emb)) if chem_embeddings_test.size else chem_emb
                text_embeddings_test = np.concatenate(
                    (text_embeddings_test, text_emb)) if text_embeddings_test.size else text_emb

        print("Test Embeddings done:", cids_test.shape, chem_embeddings_test.shape)

        emb_path = osp.join(output_path, "embeddings/")
        if not os.path.exists(emb_path):
            os.mkdir(emb_path)
        np.save(emb_path + "cids_train.npy", cids_train)
        np.save(emb_path + "cids_val.npy", cids_val)
        np.save(emb_path + "cids_test.npy", cids_test)
        np.save(emb_path + "chem_embeddings_train.npy", chem_embeddings_train)
        np.save(emb_path + "chem_embeddings_val.npy", chem_embeddings_val)
        np.save(emb_path + "chem_embeddings_test.npy", chem_embeddings_test)
        np.save(emb_path + "text_embeddings_train.npy", text_embeddings_train)
        np.save(emb_path + "text_embeddings_val.npy", text_embeddings_val)
        np.save(emb_path + "text_embeddings_test.npy", text_embeddings_test)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    new_main()