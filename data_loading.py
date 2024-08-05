import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj


def get_processed_path(args):
    file_path = {}
    in_Path = args.data

    abs_path = os.path.dirname(in_Path)

    folder_name = os.path.split(in_Path)[1].split("formatted")[0]

    file_names = [
        'train', 'val', 'test', 'train_inds', 'val_inds', 'test_inds'
    ]
    for name in file_names:
        outFile = folder_name + f"{name}.pt"
        file_path[name] = os.path.join(abs_path, outFile)
    return file_path


def get_data(args, data_config):
    '''Loads the AML transaction data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''
    file_path = get_processed_path(args)
    if os.path.exists(file_path["train"]) and os.path.exists(file_path["val"]) and os.path.exists(file_path["test"]) and os.path.exists(file_path["train_inds"]) and os.path.exists(file_path["val_inds"]) \
        and os.path.exists(file_path["test_inds"]):
        logging.info(f"Using the local formatted data of {args.data}")
        tr_data = torch.load(file_path["train"])
        val_data = torch.load(file_path["val"])
        te_data = torch.load(file_path["test"])
        tr_inds = torch.load(file_path["train_inds"])
        val_inds = torch.load(file_path["val_inds"])
        te_inds = torch.load(file_path["test_inds"])
        
        if args.reverse_mp:
            tr_data = create_hetero_obj(tr_data.x, tr_data.y,
                                        tr_data.edge_index, tr_data.edge_attr,
                                        tr_data.timestamps, args)
            val_data = create_hetero_obj(val_data.x, val_data.y,
                                         val_data.edge_index,
                                         val_data.edge_attr,
                                         val_data.timestamps, args)
            te_data = create_hetero_obj(te_data.x, te_data.y,
                                        te_data.edge_index, te_data.edge_attr,
                                        te_data.timestamps, args)
                                        
        return tr_data, val_data, te_data, tr_inds, val_inds, te_inds
        
    else:
        # transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv" #replace this with your path to the respective AML data objects
        transaction_file = args.data  #replace this with your path to the respective AML data objects
        df_edges = pd.read_csv(transaction_file)

        logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

        df_edges[
            'Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

        max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
        df_nodes = pd.DataFrame({
            'NodeID': np.arange(max_n_id),
            'Feature': np.ones(max_n_id)
        })
        timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
        y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

        logging.info(
            f"Illicit ratio = {y.sum()} / {len(y)} = {y.sum() / len(y) * 100:.2f}%"
        )
        logging.info(
            f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}"
        )
        logging.info(f"Number of transactions = {df_edges.shape[0]}")

        edge_features = [
            'Timestamp', 'Amount Received', 'Received Currency',
            'Payment Format'
        ]
        node_features = ['Feature']

        logging.info(f'Edge features being used: {edge_features}')
        logging.info(
            f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)'
        )

        x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
        edge_index = torch.LongTensor(
            df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
        edge_attr = torch.tensor(
            df_edges.loc[:, edge_features].to_numpy()).float()

        del df_edges, df_nodes

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        logging.info(
            f'number of days and transactions in the data: {n_days} days, {n_samples} transactions'
        )

        #data splitting
        daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [
        ]  #irs = illicit ratios, inds = indices, trans = transactions
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_irs.append(y[day_inds].float().mean())
            weighted_daily_irs.append(y[day_inds].float().mean() *
                                      day_inds.shape[0] / n_samples)
            daily_inds.append(day_inds)
            daily_trans.append(day_inds.shape[0])

        split_per = [0.6, 0.2, 0.2]
        daily_totals = np.array(daily_trans)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i, j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [
                    d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()
                ]
                split_totals_sum = np.sum(split_totals)
                split_props = [v / split_totals_sum for v in split_totals]
                split_error = [
                    abs(v - t) / t for v, t in zip(split_props, split_per)
                ]
                score = max(split_error)  #- (split_totals_sum/total) + 1
                split_scores[(i, j)] = score
            else:
                continue

        i, j = min(split_scores, key=split_scores.get)
        #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [
            list(range(i)),
            list(range(i, j)),
            list(range(j, len(daily_totals)))
        ]
        logging.info(f'Calculate split: {split}')

        #Now, we seperate the transactions based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                split_inds[i].append(
                    daily_inds[day]
                )  #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

        tr_inds = torch.cat(split_inds[0])
        val_inds = torch.cat(split_inds[1])
        te_inds = torch.cat(split_inds[2])

        del split_inds, split_scores, split_props, split_error, I, d_ts, split_totals
        del daily_inds, daily_irs, daily_totals, daily_trans
        
        logging.info(
            f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0]}"
        )
        logging.info(
            f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1]}")
        logging.info(
            f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2]}")

        #Creating the final data objects
        logging.info("Creating tr_data.")
        tr_x = x
        e_tr = tr_inds.numpy()
        tr_edge_index, tr_edge_attr, tr_y, tr_edge_times = edge_index[:, e_tr], edge_attr[
            e_tr], y[e_tr], timestamps[e_tr]
        tr_data = GraphData(x=tr_x,
                            y=tr_y,
                            edge_index=tr_edge_index,
                            edge_attr=tr_edge_attr,
                            timestamps=tr_edge_times)
        logging.info("Adding ports for tr_data.")
        tr_data.add_ports()
        tr_data.x = z_norm(tr_data.x)
        tr_data.edge_attr = z_norm(tr_data.edge_attr)
        logging.info(f'train data object: {tr_data}')
        torch.save(tr_data, file_path["train"])
        torch.save(tr_inds, file_path["train_inds"])
        del tr_data, tr_edge_index, tr_edge_attr, tr_edge_times, tr_x, tr_y, e_tr
        
        logging.info("Creating val_data.")
        val_x = x
        e_val = np.concatenate([tr_inds, val_inds])
        val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:, e_val], edge_attr[
            e_val], y[e_val], timestamps[e_val]
        val_data = GraphData(x=val_x,
                            y=val_y,
                            edge_index=val_edge_index,
                            edge_attr=val_edge_attr,
                            timestamps=val_edge_times)
        logging.info("Adding ports for val_data.")
        val_data.add_ports()
        val_data.x = z_norm(val_data.x)
        val_data.edge_attr = z_norm(val_data.edge_attr)
        logging.info(f'val data object: {val_data}')
        torch.save(val_data, file_path["val"])
        torch.save(val_inds, file_path["val_inds"])
        del val_data, val_inds, val_edge_index, val_edge_attr, val_edge_times, val_x, val_y, e_val
        del tr_inds
        
        logging.info("Creating te_data.")
        te_x = x
        te_edge_index, te_edge_attr, te_y, te_edge_times = edge_index, edge_attr, y, timestamps
        te_data = GraphData(x=te_x,
                            y=te_y,
                            edge_index=te_edge_index,
                            edge_attr=te_edge_attr,
                            timestamps=te_edge_times)
        logging.info("Adding ports for te_data.")
        te_data.add_ports()
        te_data.x = z_norm(te_data.x)
        te_data.edge_attr = z_norm(te_data.edge_attr)
        logging.info(f'test data object: {te_data}')
        torch.save(te_data, file_path["test"])
        torch.save(te_inds, file_path["test_inds"])
        del te_data, te_inds, te_edge_index, te_edge_attr, te_edge_times, te_x, te_y
        
        del edge_attr, edge_index, timestamps, x, y

        tr_data = torch.load(file_path["train"])
        val_data = torch.load(file_path["val"])
        te_data = torch.load(file_path["test"])
        tr_inds = torch.load(file_path["train_inds"])
        val_inds = torch.load(file_path["val_inds"])
        te_inds = torch.load(file_path["test_inds"])
        
        if args.reverse_mp:
            logging.info("Converting tr_data to Hetero")
            tr_data = create_hetero_obj(tr_data.x, tr_data.y,
                                        tr_data.edge_index, tr_data.edge_attr,
                                        tr_data.timestamps, args)
            logging.info("Converting val_data to Hetero")
            val_data = create_hetero_obj(val_data.x, val_data.y,
                                         val_data.edge_index,
                                         val_data.edge_attr,
                                         val_data.timestamps, args)
            logging.info("Converting te_data to Hetero")
            te_data = create_hetero_obj(te_data.x, te_data.y,
                                        te_data.edge_index, te_data.edge_attr,
                                        te_data.timestamps, args)
        
        return tr_data, val_data, te_data, tr_inds, val_inds, te_inds
