from odkd.models.backbone import mobilenet_v2, vgg16_bn
from odkd.models.ssdlite import SSDLite

def create_ssdlite(backbone, config, **kwargs):
    if backbone.lower() == 'vgg16':
        backbone = vgg16_bn().features
    elif backbone.lower() == 'mobilenetv2':
        backbone = mobilenet_v2().features
    else:
        raise ValueError('Unsupport backbone %s' % backbone)

    args = config[SSDLite.__init__.__code__.co_varnames[1:SSDLite.__init__.__code__.co_argcount]]
    args.update(kwargs)
    return SSDLite(backbone, **args)
