import json
import logging
import os
import sys
from typing import Any

import loguru


def serialize(record: dict[str, Any]) -> str:
    """
    Serialize a log record into a JSON-formatted string.

    Args:
        record (dict): A dictionary representing the log record.

    Returns:
        str: JSON-formatted log entry.
    """
    extra_vars = record.get("extra", {})
    process_name = extra_vars.pop("process", None)

    default_dict = {
        "level": record["level"].name,
        "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "message": record["message"],
        **extra_vars,
    }

    if process_name:
        process_dict = {"process": process_name}
        process_dict.update(default_dict)
        return json.dumps(process_dict)

    return json.dumps(default_dict)


def create_log_entry(record: dict[str, Any]) -> str:
    """
    Create a log entry as a formatted string.

    Args:
        record (dict): A dictionary representing the log record.

    Returns:
        str: Formatted log entry.
    """
    extra_vars = record.get("extra", {})
    process_name = extra_vars.pop("process", None)

    timestamp = record["time"].strftime("%Y-%m-%d %H:%M:%S")
    level = record["level"].name
    message = record["message"]

    if process_name:
        return f"{timestamp} [{level}] ({process_name}): {message}"
    else:
        return f"{timestamp} [{level}]: {message}"
    

def patching(record):
    # record["extra"]["serialized"] = serialize(record)
    record["extra"]["formatted"] = create_log_entry(record)


def get_logger(
    log_process: str = None, log_path: str | None = sys.stderr
) -> logging.Logger:
    """
    Creates a log file and returns a Logger object.

    Args:
        log_process (str): Name of the current process.
        log_path (Optional[str]): Path to the log file. If specified, logs will be saved to this file.
            If not specified, logs will be printed to the console.

    Returns:
        logging.Logger: Logger object that will log information and warnings, etc.
    """

    logger = loguru.logger
    logger.remove()  # Remove default handler as we will be adding it below

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logger = logger.patch(patching)

    # Bind logger so the defined process now prints to logs by default
    logger = logger.bind(process=log_process)

    logger.add(log_path, level=log_level, format="{extra[formatted]}")

    return logger


if __name__ == "__main__":
    my_logger = get_logger(log_process="MyApp", log_path="my_app.log")
    my_logger.info("Process1")
    my_logger.debug("Hello", product_id="123")
