"""
The Adversarial Robustness Toolbox (ART).
"""
import logging.config

# Project Imports
from tests import attacks
from tests import defences
from tests import estimators
from tests import evaluations
from tests import metrics
from tests import preprocessing

# Semantic Version
__version__ = "1.7.2"

# pylint: disable=C0103

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.NullHandler",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "art": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "INFO", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
