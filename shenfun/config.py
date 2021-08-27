import os
import yaml

# The configuration file can be overloaded locally or in '~/.shenfun'

locations = [os.path.dirname(__file__),
             os.path.expanduser('~/.shenfun'),
             os.getcwd()]

config = {}
for loc in locations:
    fl = os.path.join(loc, 'shenfun.yaml')
    try:
        with open(fl, 'r') as yf:
            config.update(yaml.load(yf, Loader=yaml.FullLoader))
        yf.close()
    except FileNotFoundError:
        pass
