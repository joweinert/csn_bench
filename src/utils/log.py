import logging
import sys
import time
import psutil
import os
from contextlib import contextmanager

BANNER_LEVEL = 25
logging.addLevelName(BANNER_LEVEL, "BANNER")

CSN_BANNER_ART = r"""
  _____   _____   _   _ 
 / ____| / ____| | \ | |
| |     | (___   |  \| |
| |      \___ \  | . ` |
| |____  ____) | | |\  |
 \_____||_____/  |_| \_|
"""

class CustomFormatter(logging.Formatter):
    """
    Custom formatter: Colors for console, standard format for file.
    """
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Standard format: [Time] - [Level] - [Message]
    format_str = "%(asctime)s - %(levelname)-8s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        BANNER_LEVEL: cyan + "%(message)s" + reset,  # Banners are Cyan, no timestamp
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)

def setup_logger(name="CSN", log_file=None, level=logging.INFO):
    """
    Initializes a logger that outputs to both console (colored) and file (clean).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        #no ansi for file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger

# --- Helper Functions ---

def log_startup_banner(logger):
    """Prints the CSN ASCII art at the start of the run."""
    logger.log(BANNER_LEVEL, CSN_BANNER_ART)
    logger.info("Initializing CSN Research Pipeline...")
    log_system_resources(logger)

def log_section(logger, title):
    """Prints a separator and title for major experiment phases."""
    line = "-" * 60
    logger.log(BANNER_LEVEL, f"\n{line}\n{title.center(60)}\n{line}")

def log_system_resources(logger):
    """Logs current CPU and RAM usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # bytes to GB
    ram_gb = mem_info.rss / (1024 ** 3)
    cpu_pct = psutil.cpu_percent(interval=None)
    
    logger.info(f"[System] CPU: {cpu_pct:.1f}% | RAM: {ram_gb:.2f} GB used")

@contextmanager
def log_step(logger, description, log_resources=False):
    """
    Context manager to time a block of code.
    Usage:
        with log_step(logger, "Training VAE"):
            ...
    """
    logger.info(f"START: {description}")
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"DONE: {description} ({duration:.4f}s)")
        
        if log_resources:
            log_system_resources(logger)
