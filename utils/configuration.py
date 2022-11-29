import commentjson
import yaml


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return commentjson.load(json_file)
    else:
        return config_file


def validate_config(config_file):
    config = parse_configuration(config_file)


def describe_config(config_file):
    config = parse_configuration(config_file)
    print('Configuration:')
    print(yaml.dump(config, allow_unicode=True, indent=4))
