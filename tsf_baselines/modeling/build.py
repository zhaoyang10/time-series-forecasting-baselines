from .informer.model import Informer, InformerStack

def get_model_class(model_name):
    """Return the algorithm class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Model not found: {}".format(model_name))
    return globals()[model_name]

def build_network(cfg):

    if cfg.MODEL.NAME in ['Informer', 'Informerstack']:
        model_dict = {
            'Informer': Informer,
            'Informerstack': InformerStack,
        }
        _model = get_model_class(cfg.MODEL.NAME)
        e_layers = cfg.MODEL.E_LAYERS if cfg.MODEL.NAME == 'informer' else cfg.MODEL.E_LAYERS

        model = _model(
            cfg.MODEL.ENC_IN,
            cfg.MODEL.DEC_IN,
            cfg.MODEL.C_OUT,
            cfg.MODEL.SEQ_LEN,
            cfg.MODEL.LABEL_LEN,
            cfg.MODEL.PRED_LEN,
            cfg.MODEL.FACTOR,
            cfg.MODEL.D_MODEL,
            cfg.MODEL.N_HEADS,
            e_layers, #cfg.MODEL.E_LAYERS,
            cfg.MODEL.D_LAYERS,
            cfg.MODEL.D_FF,
            cfg.MODEL.DROP_OUT,
            cfg.MODEL.ATTN,
            cfg.MODEL.EMBED,
            cfg.MODEL.FREQ,
            cfg.MODEL.ACTIVATION,
            cfg.MODEL.OUTPUT_ATTENTION,
            cfg.MODEL.DISTIL,
            cfg.MODEL.MIX
        ).float()

    return model