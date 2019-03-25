import configparser


def idx1 (x):
    return x[1]


def abs_idx1 (x):
    return abs(x[1])


def get_config_val(section, key):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config[section][key]
