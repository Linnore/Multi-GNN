import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data
from training import train_gnn
from inference import infer_gnn
import json

from AMLworld import AMLworld
from data_loading_aml import data_loading_aml


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()

    if args.ibm_data_loading:
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    else:
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = data_loading_aml(args)

    # tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        #Training
        logging.info(f"Running Training")
        train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args,
                  data_config)
    else:
        #Inference
        logging.info(f"Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args,
                  data_config)

if __name__ == "__main__":
    main()
