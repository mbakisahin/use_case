import logging

def setup_logging():
    """
    Configure the logging settings.

    Sets the logging level to INFO and specifies the format.
    Logs are written to both a file and the console.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("../../project.log"),
                            logging.StreamHandler()
                        ])

def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
