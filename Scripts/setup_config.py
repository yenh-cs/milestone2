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
    utd_p = os.path.abspath(utd_data_path)

    config = ConfigParser()
    config['UTD'] = {
        "path": utd_p
    }
    file_dir = os.path.abspath(os.path.join(__file__, "../.."))
    save_p = os.path.join(
        file_dir,
        '.config.ini'
    )
    with open(save_p, 'w') as f:
        config.write(f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="setup UTD")
    parser.add_argument("utd_data_path", type=str, help="path to UTD directory")
    args = parser.parse_args()

    setup_utd(args.utd_data_path)
