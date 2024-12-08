import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'ckp', path)

print(get_data('ckp/GNNT-ckp-0dc3-03-1kk-weights.pt'))