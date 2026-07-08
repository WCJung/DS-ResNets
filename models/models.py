from models import ResNets as resnet


def build_ds_resnet(layers, n_class, use_avgpool=False, use_50176=False):
    """차원 보존(DS) ResNet 백본 생성.

    layers 예: [2, 2, 2, 2] (8 블록, ResNet-18 대응)
               [3, 4, 6, 3] (16 블록, ResNet-50 대응)
    """
    return resnet.ResNet(block=resnet.Bottleneck, layers=layers,
                         num_classes=n_class,
                         use_avgpool=use_avgpool, use_50176=use_50176)
