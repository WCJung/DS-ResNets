from models import ResNets as resnet
from torch import nn


def model_(params, device, use_avgpool=False, use_50176=False, layers=None):
    par = dict()
    par['model'] = params.model
    par['weight'] = params.weight

    # layers 우선순위: 함수 인자 > argparse --bc
    if layers is None:
        layers = par.get('block', [3, 4, 6, 3])

    mod = resnet.ResNet(block=resnet.Bottleneck, layers=layers,
                        use_avgpool=use_avgpool, use_50176=use_50176)
    if par['weight']:
        raise NotImplementedError("Pretrained weights not configured — run with --weight False")

    return mod

def transfer(model, seed, n_class, feat=None):
    # feat=None のとき: model.fc の in_features をそのまま使う
    if feat is None:
        feat = model.fc.in_features
    model.fc = nn.Linear(feat, n_class)

