import os
from configparser import ConfigParser

def setup_utd(utd_data_path: str):
    """
    creates a config file and saves
    Args:
        root_path: path to

    Returns:
        None, saves a config file
    """
    config = ConfigParser()
    config['UTD'] = {
        "path": os.path.abspath(utd_data_path)
    }

    file_p = os.path.abspath(__file__)
    save_p = os.path.join(
        os.path.dirname(file_p),
        'config.ini'
    )

    with open(save_p, 'w') as f:
        config.write(f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="setup UTD")
    parser.add_argument("utd_data_path", type=str, help="path to UTD directory")
    args = parser.parse_args()

    setup_utd(args.utd_data_path)
