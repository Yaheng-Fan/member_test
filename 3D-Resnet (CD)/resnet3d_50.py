import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchsummary import summary
from torchvision import models
import os

# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm3d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(places),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(places),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck_addBN(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck_addBN, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(places),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(places),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm3d(places*self.expansion)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.BN(out)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        feature = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc(feature)
        x = self.sigmoid(x)
        return x#, feature

class ResNet_addBN(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet_addBN, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck_addBN(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck_addBN(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        feature = x

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        x = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class ResNet_2fc(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet_2fc, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(2048, 10)
        self.fc2 = nn.Linear(10, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        feature = x

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        x = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class ResNet_2fc_5(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet_2fc_5, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(2048, 5)
        self.fc2 = nn.Linear(5, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        feature = x

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        x = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc1(x)
        feature = x
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class ResNet_2fc_1024(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet_2fc_1024, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        feature = x

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        x = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class ResNet_feature_extract(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet_feature_extract, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_classes)
        # self.fc1 = nn.Linear(2048, 5)
        # self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        feature = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc(feature)
        # feature = self.fc1(x)
        # x = self.fc2(feature)
        x = self.sigmoid(x)
        return x, feature

class ResNet_feature_extract__(nn.Module):
    def __init__(self, blocks, num_classes=1, expansion=4):
        super(ResNet_feature_extract__, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_l5yer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool3d(2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape:", x.shape)
        x = self.conv1(x)
        # print("con1 shape:", x.shape)

        x = self.layer1(x)
        # print("layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("layer4 shape:", x.shape)

        feature = x

        x = self.avgpool(x)
        # print("avgpool shape:", x.shape)

        # x = nn.Dropout(0.5)(x)

        feature = x.view(x.size(0), -1)
        # print("view shape:", x.shape)
        x = self.fc(feature)
        x = self.sigmoid(x)
        return x, feature

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet_middle():
    return ResNet([3, 4, 16, 3])

def ResNet_high_middle():
    return ResNet([3, 4, 30, 3])

def ResNet_high_middle_m():
    return ResNet([3, 6, 30, 3])

def ResNet101_m():
    return ResNet([3, 6, 23, 3])

def ResNet101_m_2fc():
    return ResNet_2fc([3, 6, 23, 3])

def ResNet101_m_2fc_5():
    return ResNet_2fc_5([3, 6, 23, 3])

def ResNet101_m_2fc_1024():
    return ResNet_2fc_1024([3, 6, 23, 3])

def ResNet101_m_feature():
    return ResNet_feature_extract([3, 6, 23, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])

def resnet1():
    return ResNet([1, 1, 1, 1])

def Resnet101_m_addBN():
    return ResNet_addBN([3, 6, 23, 3])


if __name__=='__main__':
    # models = torchvision.models.resnet50()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = ResNet101_m_feature().to(device)
    # net = resnet1().to(device)
    # summary(net, (1, 32, 128, 128))
    # models = ResNet101_m()
    # for name, module in net._modules['layer4']._modules.items():
    #     if name == '2':
    #         print(module)

    # print(models)
    #
    # params = list(net.parameters())  # 将参数变换为列表
    # for num, temp in net.named_parameters():
    #     print(num, temp.shape)
    import heapq
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = ResNet101_m()
    for m in range(1, 11):
        model.load_state_dict(torch.load("/media/ds/新加卷/LZX_Crohn/new_result/exp2_159train_noaug2/final_model/"+str(m)+".pkl"))  # 加载模型
        for layer in model.modules():
            if(isinstance(layer, nn.Linear)):
                weight_tensor = layer.weight
                weight_np = weight_tensor.detach().numpy()
                weight_list = []
                for i in range(2048):
                    weight_list.append(abs(weight_np[0, i]))
                re = map(weight_list.index, heapq.nlargest(10, weight_list))
                print("model_"+str(m), list(re))