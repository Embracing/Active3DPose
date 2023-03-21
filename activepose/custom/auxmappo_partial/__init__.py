from ray.rllib.models import ModelCatalog

from .model import AuxRNNMAPartialModel


ModelCatalog.register_custom_model('aux_rnn_ma_partial', AuxRNNMAPartialModel)
