import inspect
from . import STRATEGIES


def get_strategy(strategy_name: str, **kwargs):

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_class = STRATEGIES[strategy_name]
    strategy_args = inspect.getfullargspec(strategy_class.__init__).args[1:]

    validated_kwargs = {}
    for arg in strategy_args:
        if arg in kwargs:
            validated_kwargs[arg] = kwargs[arg]

    return strategy_class(**validated_kwargs)
