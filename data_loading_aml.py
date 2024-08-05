from AMLworld import AMLworld

def data_loading_aml(arg,*args,**kwargs):
    tr_data = AMLworld("./aml_data/Small_HI", split="train")
    tr_data = tr_data[0]
    tr_inds = tr_data.tr_inds
    val_data = AMLworld("./aml_data/Small_HI", split="val")
    val_data = val_data[0]
    val_inds = val_data.val_inds
    te_data = AMLworld("./aml_data/Small_HI", split="test")
    te_data = te_data[0]
    te_inds = te_data.te_inds

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds
