import json
import logging
import os

logger = logging.getLogger('configs')
logging.basicConfig(level=logging.INFO)


def configs():
    if not os.path.exists('configs.json'):
        logger.warning(f"The configs.json file does not exist!")
        return
    with open('configs.json') as f:
        return json.load(f)
