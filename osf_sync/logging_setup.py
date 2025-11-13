import logging, os, json, sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s | extras=%(extras)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    # default extras
    logger = logging.LoggerAdapter(logger, extra={"extras": "{}"})
    return logger

def with_extras(logger: logging.Logger, **extras):
    # attach JSON extras for consistent structured logs
    return logging.LoggerAdapter(logger.logger if hasattr(logger, "logger") else logger,
                                 extra={"extras": json.dumps(extras, ensure_ascii=False)})
