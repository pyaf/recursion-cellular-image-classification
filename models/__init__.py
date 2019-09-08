from .pretrained import *
from .zoominnet import ZoomInNet


def get_model(model_name, num_classes=1, pretrained="imagenet"):

    if model_name == "resnext101_32x16d":
        return resnext101_32x16d(num_classes)

    elif model_name.startswith("efficientnet"):
        return efficientNet(model_name, num_classes, pretrained)

    elif model_name == 'zoominnet':
        model = ZoomInNet()
        return model

    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    for params in model.parameters():
        params.requires_grad = False

    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=True
    )
    return model


if __name__ == "__main__":
    # model_name = "se_resnext50_32x4d_v2"
    # model_name = "nasnetamobile"
    # model_name = "resnext101_32x4d_v0"
    model_name = "efficientnet-b5"
    classes = 1
    size = 256
    model = get_model(model_name, classes, "imagenet")
    image = torch.Tensor(
        3, 3, size, size
    )  # BN layers need more than one inputs, running mean and std
    # image = torch.Tensor(1, 3, 112, 112)
    # image = torch.Tensor(1, 3, 96, 96)

    output = model(image)
    print(output.shape)
    pdb.set_trace()


""" footnotes

[1]: model.avgpool is already AdapativeAvgPool2d, and model's forward method handles flatten and stuff. So here I'm just adding a trainable the last fc layer, after few epochs the model's all layers will be set required_grad=True
Apart from that this model is trained on instagram images, remove imagenet mean and std, only gotta divide by 255, so mean=0,std=1

[2]: efficientnet models are trained on imagenet, so make sure mean and std are of imagenet.
"""
