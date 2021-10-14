"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from .tsm_util import tsm
from .selfy import SELFYBlock
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments,stride=1, downsample=None, remainder=0):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remainder =remainder
        self.num_segments = num_segments        

    def forward(self, x):
        identity = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out        
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder
        self.num_segments = num_segments

    def forward(self, x):
        identity = x  
        out = tsm(x, self.num_segments, 'zero')   
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,num_segments, stride=1, downsample=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder        
        self.num_segments = num_segments        

    def forward(self, x):
        identity = x
        out = tsm(x, self.num_segments, 'zero')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):

    def __init__(self, block, block2, layers, num_segments, flow_estimation, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()          
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax = nn.Softmax(dim=1)        
        self.num_segments = num_segments     
        self.flow_estimation = flow_estimation
        
        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
        
        ## SELFY
        ## Hard-coded hyperparameters of SELFY module
        if flow_estimation:
            self.selfy = SELFYBlock(
                128*block.expansion,
                128,
                num_segments=num_segments,
                window=[5,9,9],
                ext_chnls=[4,16,64,64],
                int_chnls=[64,64,64,64]
            )
        
        self.layer3 = self._make_layer(block, 256, layers[2],  num_segments=num_segments, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],  num_segments=num_segments, stride=2)       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv1d(512*block.expansion, num_classes, kernel_size=1, stride=1, padding=0,bias=True)         
        
        for m in self.modules():       
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, SELFYBlock):
                    nn.init.constant_(m.stss_integration.upsample[-2].weight, 0)

        
    def _make_layer(self, block, planes, blocks, num_segments, stride=1):       
        downsample = None        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
            
        return nn.Sequential(*layers)            
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)                             
        x = self.layer2(x)          
        
        # SELFY Block
        if (self.flow_estimation == 1):  
            x = self.selfy(x)

        x = self.layer3(x)                           
        x = self.layer4(x)
        x = self.avgpool(x)    
        x = x.view(x.size(0), -1,1)    
                       
        x = self.fc1(x)      
        return x


def resnet18(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(BasicBlock, BasicBlock, [2, 2, 2, 2], num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)  
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
        model.load_state_dict(new_state_dict)
    return model


def resnet34(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0,**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):
        model = ResNet(BasicBlock, BasicBlock, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation,  **kwargs)        
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
        model.load_state_dict(new_state_dict)
    return model


def resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
        model.load_state_dict(new_state_dict)
    return model


def resnet101(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 23, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
        model.load_state_dict(new_state_dict)
    return model
