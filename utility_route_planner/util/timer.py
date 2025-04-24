# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import time
import structlog

from settings import Config

logger = structlog.get_logger(__name__)


def time_function(func):
    def wrapper(*args, **kwargs):
        if Config.DEBUG:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"*** Function {func.__name__} took {end_time - start_time:.2f} seconds.")
        else:
            return func(*args, **kwargs)
        return result

    return wrapper
