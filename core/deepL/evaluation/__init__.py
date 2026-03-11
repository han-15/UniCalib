from easydict import EasyDict as edict

evaluations = {}

def register_evaluation(cls):
    evaluations[cls.__name__] = cls
    return cls

def get_evaluation(name: str, cfg: edict):
    assert name in evaluations, f"evaluation {name} is not registered"
    return evaluations[name](cfg)
