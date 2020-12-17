import json
import logging
import os

logger = logging.getLogger('configs')
logging.basicConfig(level=logging.INFO)


def configs(config_file):
    if not os.path.exists(config_file):
        logger.warning(f"The {config_file} file does not exist!")
        return
    with open(config_file) as f:
        return json.load(f)
