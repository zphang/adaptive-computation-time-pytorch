from act.configuration import Config
from act.train import train
from act.models import resolve_model
from act.data import resolve_data_manager


if __name__ == "__main__":
    config = Config.parse_configuration()
    train(
        config=config,
        model=resolve_model(config),
        data_manager=resolve_data_manager(config)
    )
