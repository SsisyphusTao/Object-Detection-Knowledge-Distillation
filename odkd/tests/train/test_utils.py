from odkd.interface import create_optimizer, create_scheduler


def test_create_scheduler(config, ssdlite):
    config['params'] = ssdlite.parameters()
    optimizer = config['optimizer'] = create_optimizer(config)
    scheduler = create_scheduler(config)
    optimizer.step()
    scheduler.step()
