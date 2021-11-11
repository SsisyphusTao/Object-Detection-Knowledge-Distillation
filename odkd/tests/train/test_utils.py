from odkd.train._utils import create_optimizer, create_scheduler


def test_create_scheduler(config, ssdlite):
    optimizer = create_optimizer(config)(ssdlite.parameters())
    scheduler = create_scheduler(config)(optimizer)
    optimizer.step()
    scheduler.step()
