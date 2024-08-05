import logging

from AMLworld import AMLworld
from data_util import create_hetero_obj

def data_loading_aml(args):
    if args.root is None:
        raise ValueError
    tr_data = AMLworld(args.root, split="train", verbose=False)
    tr_data = tr_data[0]
    tr_inds = tr_data.tr_inds
    val_data = AMLworld(args.root, split="val", verbose=False)
    val_data = val_data[0]
    val_inds = val_data.val_inds
    te_data = AMLworld(args.root, split="test", verbose=False)
    te_data = te_data[0]
    te_inds = te_data.te_inds

    if args.reverse_mp:
        logging.info("Converting tr_data to Hetero")
        tr_data = create_hetero_obj(tr_data.x, tr_data.y, tr_data.edge_index,
                                    tr_data.edge_attr, tr_data.timestamps,
                                    args)
        logging.info("Converting val_data to Hetero")
        val_data = create_hetero_obj(val_data.x, val_data.y,
                                     val_data.edge_index, val_data.edge_attr,
                                     val_data.timestamps, args)
        logging.info("Converting te_data to Hetero")
        te_data = create_hetero_obj(te_data.x, te_data.y, te_data.edge_index,
                                    te_data.edge_attr, te_data.timestamps,
                                    args)

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds
