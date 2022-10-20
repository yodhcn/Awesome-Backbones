import torch
# from configs.backbones.resnet import ResNet
from configs.backbones.resnet_cbam import ResNetCBAM

backbone = ResNetCBAM(
    depth=50,
    num_stages=4,
    out_indices=(3, ),
    style='pytorch',
)
print(backbone)

inputs = torch.zeros(1, 3, 224, 224)
print(inputs.shape)
outputs = backbone(inputs)
print(outputs[0].shape)
