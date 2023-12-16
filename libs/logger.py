import logging

class Logger:
    _logger = None

    @staticmethod
    def get_logger(file_name):
        if Logger._logger is None:
            Logger._logger = Logger._setup_logger(file_name)
        return Logger._logger

    @staticmethod
    def _setup_logger(file_name):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        # Create a file handler
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Create a stream handler to print log messages on the console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger
