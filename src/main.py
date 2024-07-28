# encoding: utf-8
import argparse
import itertools
import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
import pickle

import dgl
import numpy as np
import torch
# smc
torch.backends.cudnn.enabled = False
import yaml
from tqdm import tqdm
import random
from yaml import SafeLoader
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_all_graph, build_sub_graph_0
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import *
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import scipy.sparse as sp

from src.hgls.load_data import load_data
from src.hgls.utils import create_data, Collate
from src.hgls.utils_new import myFloder_new


# 测试
def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name,
         static_graph, time_list, history_time_nogt, mode, co, node_id_new, s_t, test_set):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name,
                                                                  checkpoint['epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    # input_list = [snap for snap in history_list[-args.test_history_len:]]
    # g_list = build_all_graph(num_nodes, num_rels, input_list, use_cuda, args.gpu)

    if args.multi_step:
        all_tail_seq = sp.load_npz(
            '../data/{}/history/tail_history_{}.npz'.format(args.dataset, history_time_nogt))
        # rel
        all_rel_seq = sp.load_npz(
            '../data/{}/history/rel_history_{}.npz'.format(args.dataset, history_time_nogt))

    # smc
    all_list = history_list + test_list
    length = len(history_list)
    # input_list = [snap for snap in history_list[-args.test_history_len:]]

    # smc  背景图
    g_list = build_all_graph(num_nodes, num_rels, all_list[0:args.l_length], use_cuda, args.gpu)    # shit
    # /smc

    # for time_idx, test_snap in enumerate(tqdm(test_list[0:len(test_list)-1])):
    #     time_idx = time_idx + 1
    #     test_snap = test_list[time_idx]
    for time_idx, test_snap in enumerate(tqdm(test_list[0:len(test_list)-1])):   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # smc
        input_list = [snap for snap in
                      all_list[len(history_list) + time_idx - args.test_history_len:len(history_list) + time_idx]]

        # smc 构造时空同步图
        history_glist_0 = build_sub_graph_0(num_nodes, num_rels, input_list, use_cuda, args.gpu)
        # /smc

        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]

        # data_list = co.collate_rel([[[test_snap], length + time_idx]])
        # data_list = {}
        # data_list['t'] = [5]
        data_list = test_set.get_data(time_idx)   #！！！！！！！！！！！！！！！！！！！！！
        # data_list = test_set.get_data(time_idx-1)

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)

        # smc 构造超图神经网络要使用的图数据信息
        index_list = []
        index = np.concatenate(all_list[0:args.l_length])
        index = torch.from_numpy(index[:, [0, 1, 2]])
        inverse_index = index[:, [2, 1, 0]]
        inverse_index[:, 1] = inverse_index[:, 1] + num_rels
        index = torch.cat([index, inverse_index]).transpose(1, 0)
        index_list.append(index[[0, 1], :])  # u, r
        # /smc

        # # smc 构造超图神经网络要使用的图数据信息   时序超图使用的
        # index_list = []
        # for index in input_list:
        #     index = torch.from_numpy(index[:, [0, 1, 2]])
        #     inverse_index = index[:, [2, 1, 0]]
        #     inverse_index[:, 1] = inverse_index[:, 1] + num_rels
        #     index = torch.cat([index, inverse_index]).transpose(1, 0)
        #     index_list.append(index[[0, 1], :])  # u, r
        # # /smc

        # get history
        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data = histroy_data.cpu().numpy()
        if args.multi_step:
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            # rel
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        else:
            all_tail_seq = sp.load_npz(
                '../data/{}/history/tail_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            # rel
            all_rel_seq = sp.load_npz(
                '../data/{}/history/rel_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

            # smc 局部答案
            all_tail_local_seq = sp.load_npz(
                '../data/{}/history_1hop_10/tail_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            seq_local_idx = histroy_data[:, 0]
            tail_local_seq = torch.Tensor(all_tail_local_seq[seq_local_idx].todense())
            one_hot_tail_local_seq = tail_local_seq.masked_fill(tail_local_seq != 0, 1)
            # /smc
        if use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.cuda()
            one_hot_rel_seq = one_hot_rel_seq.cuda()
            one_hot_tail_local_seq = one_hot_tail_local_seq.cuda()


        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph,
                                                                 test_triples_input, one_hot_tail_seq, one_hot_rel_seq,
                                                                 use_cuda, g_list, len(history_list) + time_idx,
                                                                 index_list, history_glist_0, one_hot_tail_local_seq,
                                                                 data_list, node_id_new[:, data_list['t'][0]].cuda(),
                                        (data_list['t'][0] - s_t[:, data_list['t'][0]]).cuda())

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score,
                                                                                        all_ans_r_list[time_idx],
                                                                                        eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score,
                                                                                all_ans_list[time_idx], eval_bz=1000,
                                                                                rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1

    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r


def run_experiment(args, history_len=None, n_layers=None, dropout=None, n_bases=None, angle=None, history_rate=None):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate
    mrr_raw = None
    mrr_filter = None
    mrr_raw_r = None
    mrr_filter_r = None
    hit_result_raw = None
    hit_result_filter = None
    hit_result_raw_r = None
    hit_result_filter_r = None

    # /smc hgls的加载数据与一些初始化
    num_nodes, num_rels, train_list, valid_list, test_list, total_data, all_ans_list_test, all_ans_list_r_test, \
        all_ans_list_valid, all_ans_list_r_valid, graph, node_id_new, s_t, s_f, s_l, train_sid, valid_sid, test_sid, \
        total_times, time_idx = load_data(args.dataset)
    short_con = yaml.load(open('./hgls/short_config.yaml'), Loader=SafeLoader)[args.dataset]
    long_con = yaml.load(open('./hgls/long_config.yaml'), Loader=SafeLoader)[args.dataset]
    long_con['time_length'] = len(total_data)
    long_con['time_idx'] = time_idx
    long_con['h_dim'] = args.n_hidden
    long_con['out_dim'] = args.n_hidden
    co = Collate(num_nodes, num_rels, s_f, s_t, len(total_data), args.dataset, long_con['encoder'],
                 long_con['decoder'], max_length=long_con['max_length'], all=False, graph=graph, k=2)

    # # 本地服务器
    # train_path = '/home/smc/mgesl/data/' + args.dataset + '/noleak_length_20_encoder_regcn_decoder_rgat_r1_hop_2/train/'
    # valid_path = '/home/smc/mgesl/data/' + args.dataset + '/noleak_length_20_encoder_regcn_decoder_rgat_r1_hop_2/val/'
    # test_path = '/home/smc/mgesl/data/' + args.dataset + '/noleak_length_20_encoder_regcn_decoder_rgat_r1_hop_2/test/'

    # # 本地服务器-测试集泄露
    # train_path = '/home/smc/mgesl/data/' + args.dataset + '/leak_length_10_encoder_regcn_decoder_rgat_r1_hop_2/train/'
    # valid_path = '/home/smc/mgesl/data/' + args.dataset + '/leak_length_10_encoder_regcn_decoder_rgat_r1_hop_2/val/'
    # test_path = '/home/smc/mgesl/data/' + args.dataset + '/leak_length_10_encoder_regcn_decoder_rgat_r1_hop_2/test/'

    # 曙光gpus集群
    train_path = '/public/home/detian/smc/mgesl/data/' + args.dataset + '/noleak_length_10_encoder_regcn_decoder_rgat_r1_hop_2/train/'
    valid_path = '/public/home/detian/smc/mgesl/data/' + args.dataset + '/noleak_length_10_encoder_regcn_decoder_rgat_r1_hop_2/val/'
    test_path = '/public/home/detian/smc/mgesl/data/' + args.dataset + '/noleak_length_10_encoder_regcn_decoder_rgat_r1_hop_2/test/'

    # # 曙光gpus集群-测试集泄露
    # train_path = '/public/home/detian/smc/mgesl/data/' + args.dataset + '/leak_length_50_encoder_regcn_decoder_rgat_r1_hop_2/train/'
    # valid_path = '/public/home/detian/smc/mgesl/data/' + args.dataset + '/leak_length_50_encoder_regcn_decoder_rgat_r1_hop_2/val/'
    # test_path = '/public/home/detian/smc/mgesl/data/' + args.dataset + '/leak_length_50_encoder_regcn_decoder_rgat_r1_hop_2/test/'

    train_set = myFloder_new(train_path, dgl.load_graphs)
    val_set = myFloder_new(valid_path, dgl.load_graphs)
    test_set = myFloder_new(test_path, dgl.load_graphs)
    # print(num_nodes, num_rels, s_f.shape, s_t.shape, len(total_data), args.dataset, long_con['encoder'],
    #              long_con['decoder'], long_con['max_length'], False, graph,2)


    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)  # 得到data类
    train_list, train_times = utils.split_by_time(data.train)  # 划分为snapshots，逐时间步的数据集
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    # 时间粒度
    time_interval = train_times[1] - train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)
    # 用于统计每个时间步中问题的答案有哪些，因为答案不止一个，所以再排名的时候需要过滤掉这些正确的其它答案
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "gl_rate_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp{}_{}_{}_{}-gpu{}-{}-h_noleak_C_30" \
        .format(args.history_rate, args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len,
                args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu, args.save)
    model_state_file = os.path.join('../models/', model_name)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # 是否使用静态图
    if args.add_static_graph:
        # static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        # num_static_rels = len(np.unique(static_triples[:, 1]))
        # num_words = len(np.unique(static_triples[:, 2]))
        # static_triples[:, 2] = static_triples[:, 2] + num_nodes
        # static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
        #     if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # 创建模型
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          num_static_rels,
                          num_words,
                          num_times,
                          time_interval,
                          args.n_hidden,
                          args.opn,
                          args.history_rate,
                          sequence_len=args.train_history_len,
                          num_bases=args.n_bases,
                          num_basis=args.n_basis,
                          num_hidden_layers=args.n_layers,
                          dropout=args.dropout,
                          self_loop=args.self_loop,
                          skip_connect=args.skip_connect,
                          layer_norm=args.layer_norm,
                          input_dropout=args.input_dropout,
                          hidden_dropout=args.hidden_dropout,
                          feat_dropout=args.feat_dropout,
                          aggregation=args.aggregation,
                          weight=args.weight,
                          discount=args.discount,
                          angle=args.angle,
                          use_prelearning=args.use_prelearning,
                          entity_prediction=args.entity_prediction,
                          relation_prediction=args.relation_prediction,
                          use_cuda=use_cuda,
                          gpu=args.gpu,
                          alpha=args.alpha,
                          analysis=args.run_analysis,
                          graph=graph.to(args.gpu),
                          long_con=long_con)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        print("gpu数量",torch.cuda.device_count())

    # if args.add_static_graph:
    #     static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(getModelSize(model))

    # 测试
    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
            model,
            train_list + valid_list,
            test_list,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            model_state_file,
            static_graph,
            test_times,
            history_test_time_nogt,
            "test",
            co=co,node_id_new=node_id_new,s_t=s_t, test_set=test_set)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        best_epoch = 0
        print("666777888")
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []
            # losses_show = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                # continue
                # train_sample_num = 213
                optimizer.zero_grad()  # smc
                # if train_sample_num == 0 or train_sample_num == len(idx)-1: continue
                # train_sample_num = train_sample_num + 1
                # output = train_list[train_sample_num:train_sample_num + 1]
                if train_sample_num == 0: continue                     #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                output = train_list[train_sample_num:train_sample_num + 1]   #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]
                # smc
                g_input_list = train_list[0:args.l_length]
                # smc

                # smc 构造超图神经网络要使用的图数据信息
                index_list = []
                index = np.concatenate(train_list[0:args.l_length])
                index = torch.from_numpy(index[:, [0, 1, 2]])
                inverse_index = index[:, [2, 1, 0]]
                inverse_index[:, 1] = inverse_index[:, 1] + num_rels
                index = torch.cat([index, inverse_index]).transpose(1, 0)
                index_list.append(index[[0, 1], :])  # u, r
                # /smc

                # # smc 构造超图神经网络要使用的图数据信息   时序超图使用的
                # index_list = []
                # for index in input_list:
                #     index = torch.from_numpy(index[:, [0, 1, 2]])
                #     inverse_index = index[:, [2, 1, 0]]
                #     inverse_index[:, 1] = inverse_index[:, 1] + num_rels
                #     index = torch.cat([index, inverse_index]).transpose(1, 0)
                #     index_list.append(index[[0, 1], :])  # u, r
                # # /smc

                # smc 创造hgls所需要的一条训练数据
                # print(output)
                # data_list = co.collate_rel([[output, train_sample_num]])
                # data_list = {}
                # data_list['t'] = [5]
                # print(train_sid, valid_sid, test_sid)
                # print("采样时间", train_sample_num)
                # data_list = train_set.get_data(train_sample_num-1)
                data_list = train_set.get_data(train_sample_num)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # print(train_sample_num)
                # print(data_list['sub_d_graph'])
                # 用来检验采样的子图中是否含有待预测的节点，以防止测试集泄露
                # print(node_id_new[0:5, :])
                # print(s_f[0:5, :])
                # # print("待预测的：", output[0])
                # src1, rel1, dst1 = output[0][:, [0, 1, 2]].transpose()
                # en = np.unique((src1, dst1))
                # print(en)
                # print("采样的时间：", train_sample_num)
                # print("datalist中的时间：", data_list['t'])
                # print("data_list中的采样的子图的id：", data_list['pre_d_nid'].shape)
                # print("t时间步中所有节点的id：", node_id_new[:, data_list['t'][0]].shape)
                # intersect_tensor = torch.in1d(data_list['pre_d_nid'], node_id_new[:, data_list['t'][0]])
                # print("重合的元素：",intersect_tensor)

                # /smc

                # 根据输入的历史时间戳创建历史图
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]

                # smc 构建背景图，以捕捉长期的静态历史信息
                g_list = build_all_graph(num_nodes, num_rels, g_input_list, use_cuda, args.gpu)
                # /smc

                # smc 构造时空同步图（两两时间戳为一个时空同步图）
                history_glist_0 = build_sub_graph_0(num_nodes, num_rels, input_list, use_cuda, args.gpu)
                # /smc

                # 待预测的时间戳
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [
                    torch.from_numpy(_).long() for _ in output]

                # history load 历史答案
                histroy_data = output[0]
                inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
                inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
                histroy_data = torch.cat([histroy_data, inverse_histroy_data])
                histroy_data = histroy_data.cpu().numpy()
                # tail  当前待预测的时间戳中的问题的历史答案
                all_tail_seq = sp.load_npz(
                    '../data/{}/history/tail_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                # all_tail_seq = sp.load_npz(
                #     '../data/{}/history/tail_history_{}.npz'.format(args.dataset, train_times[train_sample_num+1]))
                seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
                tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
                # rel
                all_rel_seq = sp.load_npz(
                    '../data/{}/history/rel_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                # all_rel_seq = sp.load_npz(
                #     '../data/{}/history/rel_history_{}.npz'.format(args.dataset, train_times[train_sample_num+1]))
                rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
                rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
                one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

                # smc 局部答案
                all_tail_local_seq = sp.load_npz(
                    '../data/{}/history_1hop_10/tail_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                seq_local_idx = histroy_data[:, 0]
                tail_local_seq = torch.Tensor(all_tail_local_seq[seq_local_idx].todense())
                one_hot_tail_local_seq = tail_local_seq.masked_fill(tail_local_seq != 0, 1)
                # /smc

                if use_cuda:
                    one_hot_tail_seq = one_hot_tail_seq.cuda()
                    one_hot_rel_seq = one_hot_rel_seq.cuda()
                    one_hot_tail_local_seq = one_hot_tail_local_seq.cuda()


                # 计算损失
                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph,
                                                                        one_hot_tail_seq, one_hot_rel_seq, use_cuda,
                                                                        g_list, train_sample_num, index_list, history_glist_0, one_hot_tail_local_seq,
                                                                        data_list, node_id_new[:, data_list['t'][0]].cuda(),
                                        (data_list['t'][0] - s_t[:, data_list['t'][0]]).cuda())  # 这一行都是我添加的辅助数据
                loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static #+ 0.1 * loss_show

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())
                # losses_show.append(loss_show.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print(
                "Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static),
                        best_mrr, model_name))

            # validation 验证
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
                    model,
                    train_list,
                    valid_list,
                    num_rels,
                    num_nodes,
                    use_cuda,
                    all_ans_list_valid,
                    all_ans_list_r_valid,
                    model_state_file,
                    static_graph,
                    valid_times,
                    history_val_time_nogt,
                    mode="train",
                    co=co,node_id_new=node_id_new,s_t=s_t,test_set=val_set)

                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch - best_epoch > 6:
                            print("Early Stopping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            break
                    else:
                        best_mrr = mrr_raw
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)


        # 模型训练完毕后测试
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
            model,
            train_list + valid_list,
            test_list,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            model_state_file,
            static_graph,
            test_times,
            history_test_time_nogt,
            mode="test",
            co=co,node_id_new=node_id_new,s_t=s_t,test_set=test_set)
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='TIRGN')
    parser.add_argument("--margin", type=int, default=6, help="早停")

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph", action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--use-prelearning", action='store_true', default=False,
                        help="use the prelearning graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--l-length", type=int, default=50,
                        help="prelearning graph length")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="fusion weight")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=9,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=9,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--random-grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_layer,n_hidden,l_length,alpha",
                        help="stat to use")

    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0.3,
                        help="history rate")

    parser.add_argument("--save", type=str, default="one",
                        help="number of save")

    args = parser.parse_args()

    print(args)
    if args.random_grid_search:
        out_log = '../results/{}.{}.gs'.format(args.dataset, args.encoder + "-" + args.decoder + "-" + args.save)
        o_f = open(out_log, 'w')
        print("** Random Grid Search **")
        o_f.write("** Random Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        hp_range_ = hp_range
        grid = hp_range_[hyperparameters[0]]
        random.shuffle(grid)
        for hp in hyperparameters[1:]:
            next_hp = hp_range_[hp]
            random.shuffle(next_hp)
            grid = itertools.product(grid, next_hp)
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('\n\n* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            args.test = False
            args.multi_step = False
            args.n_layer = grid_entry[0]
            args.n_hidden = grid_entry[1]
            args.l_length = grid_entry[2]
            args.alpha = grid_entry[3]
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(args)
            hits = [1, 3, 10]
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))
            # no ground truth
            args.test = True
            args.topk = 0
            args.multi_step = True
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(args)
            o_f.write("No ground truth result:\n")
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))

    # single run
    else:
        run_experiment(args)
    sys.exit()
