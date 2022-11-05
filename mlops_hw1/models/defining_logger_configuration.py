import logging.config


def get_logger():
    log_conf = {
            "version": 1,
            "formatters": {
                "simple": {
                    "format": "{asctime} - {module} - {levelname} - "
                              "{funcName}:{lineno} - {message}",
                    "datefmt": "%Y.%d.%m - %H:%M:%S",
                    "style": "{",
                },
            },
            "handlers": {
                "file_handler": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "filename": "file_handler.log",
                    "formatter": "simple",
                },
            },
            "loggers": {
                "total": {
                    "level": "DEBUG",
                    "handlers": ["file_handler"],
                },
            },
        }

    logging.config.dictConfig(log_conf)

    return logging.getLogger('total')
