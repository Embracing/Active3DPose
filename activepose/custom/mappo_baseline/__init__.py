from ray.rllib.models import ModelCatalog

from .model import MAPPOPartialModel


ModelCatalog.register_custom_model('mappo_baseline', MAPPOPartialModel)
