from importlib import import_module


NAMES = ('alfworld', 'webshop', 'scienceworld')


def load_benchmark(name, config, backend=None):
    if name not in NAMES:
        raise ValueError(f'Unknown benchmark: {name}')
    adapter = import_module(f'skillontology.benchmark.{name}.adapter').Adapter()
    environment = import_module(f'skillontology.benchmark.{name}.environment').Environment(config, backend)
    return adapter, environment
