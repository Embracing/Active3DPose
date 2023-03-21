CUSTOM_TRAINERS = {}


def load_trainers(CUSTOM_TRAINERS=CUSTOM_TRAINERS):
    from ray.tune.registry import register_trainable

    for _custom_trainer_name in CUSTOM_TRAINERS:
        _class = CUSTOM_TRAINERS[_custom_trainer_name]
        register_trainable(_custom_trainer_name, _class)


# %%
from .auxmappo_partial.model import AuxRNNMAPartialModel
from .mappo_baseline.model import MAPPOPartialModel


CUSTOM_MODELS = {
    'aux_rnn_ma_partial': AuxRNNMAPartialModel,
    'mappo_baseline': MAPPOPartialModel,
}


def load_custom_models(CUSTOM_MODELS=CUSTOM_MODELS):
    from ray.rllib.models import ModelCatalog

    for _custom_model_name in CUSTOM_MODELS:
        _class = CUSTOM_MODELS[_custom_model_name]
        ModelCatalog.register_custom_model(_custom_model_name, _class)


# %%
CUSTOM_POLICIES_FROM_MODEL_NAMES = {}
