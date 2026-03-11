from easydict import EasyDict as edict

models = {}

def register_model(cls):
    models[cls.__name__] = cls
    return cls

def create_model(cfg: edict):
    assert cfg.model.name in models, f"model {cfg.model.name} is not registered"
    return models[cfg.model.name](cfg)
