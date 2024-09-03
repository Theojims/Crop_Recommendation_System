import os
import logging

log_file = 'logging.log'
log_dir = "logs"

log_file_path = os.path.join(os.getcwd(), log_dir, log_file)

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)


logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


if __name__ == "__main__":
    logging.info('logging has started')