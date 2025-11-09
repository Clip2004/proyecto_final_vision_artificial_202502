import logging
from colorlog import ColoredFormatter
import os
 
BASE_DIR = os.path.dirname(__file__)  # sube un nivel desde /codigo_POO/
LOGS_PATH = os.path.join(BASE_DIR, "logs")
LOGS_FINAL = os.path.join(LOGS_PATH, "app.log")

class Logger:
    def __init__(self, name, log_file= LOGS_FINAL , level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.log_file = log_file
        if not self.logger.hasHandlers():
            ## Handler write in console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
 
            ## Handler write in log
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
 
            ## add handlers to logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
   
    def debug(self, msg):
        self.logger.debug(msg)
 
    def info(self, msg):
        self.logger.info(msg)
 
    def warning(self, msg):
        self.logger.warning(msg)
 
    def error(self, msg):
        self.logger.error(msg)
   
    def critical(self, msg):
        self.logger.critical(msg)
