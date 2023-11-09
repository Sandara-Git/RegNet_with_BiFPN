import torch
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import models,transforms
import matplotlib.pyplot as plt


class BottleNeck(nn.Module):
    # Scale factor of the number of output channels
    expansion = 4

    def __init__(self, in_channels, out_channels, 
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels*self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels*self.expansion,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels*self.expansion))
            

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block output
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block and stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels))


    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block ouput
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, n_blocks_list=[3, 4, 6, 3],
                 out_channels_list=[64, 128, 256, 512], num_channels=3):
        """
        Args:
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_class: number of classes for image classifcation (used in classfication head)
            n_block_lists: number of residual blocks for each conv layer (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x - conv5_x
            num_channels: the number of channels of input image
        """
        super().__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, 
                                             out_channels=64, kernel_size=7,
                                             stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2, padding=1))

        # Create four convoluiontal layers
        in_channels = 64
        # For the first block of the second layer, do not downsample and use stride=1.
        self.conv2_x = self.CreateLayer(ResBlock, n_blocks_list[0], 
                                        in_channels, out_channels_list[0], stride=1)
        
        # For the first blocks of conv3_x - conv5_x layers, perform downsampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
        # ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = self.CreateLayer(ResBlock, n_blocks_list[1], 
                                        out_channels_list[0]*ResBlock.expansion,
                                        out_channels_list[1], stride=2)
        self.conv4_x = self.CreateLayer(ResBlock, n_blocks_list[2],
                                        out_channels_list[1]*ResBlock.expansion,
                                        out_channels_list[2], stride=2)
        self.conv5_x = self.CreateLayer(ResBlock, n_blocks_list[3], 
                                        out_channels_list[2]*ResBlock.expansion,
                                        out_channels_list[3], stride=2)

        # # Average pooling (used in classification head)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # # MLP for classification (used in classification head)
        # self.fc = nn.Linear(out_channels_list[3] * ResBlock.expansion, n_classes)


    def forward(self, x):
        """
        Args: 
            x: input image
        Returns:
            C2: feature maps after conv2_x
            C3: feature maps after conv3_x
            C4: feature maps after conv4_x
            C5: feature maps after conv5_x
            y: output class
        """
        x = self.conv1(x)

        # Feature maps
        C2 = self.conv2_x(x)
        C3 = self.conv3_x(C2)
        C4 = self.conv4_x(C3)
        C5 = self.conv5_x(C4)

        # # Classification head
        # y = self.avgpool(C5)
        # y = y.reshape(y.shape[0], -1)
        # y = self.fc(y)

        return  C3, C4, C5


    def CreateLayer(self, ResBlock, n_blocks, in_channels, out_channels, stride=1):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first block of the layer.
                layer.append(ResBlock(in_channels, out_channels, 
                             stride=stride, is_first_block=True))
            else:
                # Keep the feature map size same for the rest three blocks of the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
                # ResBlock.expansion = 1 for ResNet-18, 34.
                layer.append(ResBlock(out_channels*ResBlock.expansion, out_channels))

        return nn.Sequential(*layer)


def GetFeatureMapsFromResnet(net, x):
    """
    Args:
        net: network input from torchvision.
        x: input image
    Returns:
        C2: feature maps after conv2_x
        C3: feature maps after conv3_x
        C4: feature maps after conv4_x
        C5: feature maps after conv5_x
    """
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    C2 = net.layer1(x)
    C3 = net.layer2(C2)
    C4 = net.layer3(C3)
    C5 = net.layer4(C4)
    return  C3, C4, C5


if __name__ == "__main__":
    ### Customed version ###
    # Resnet18
    #net = ResNet(BasicBlock, 1000, n_blocks_list=[2, 2, 2, 2])
    # Resnet34
    #net = ResNet(BasicBlock, 1000)
    # Resnet50
    net = ResNet(BottleNeck, 1000)
    # Resnet101
    #net = ResNet(BottleNeck, 1000, n_blocks_list=[3, 4, 23, 3])
    # Resnet152
    #net = ResNet(BottleNeck, 1000, n_blocks_list=[3, 8, 36, 3])
    # x = torch.randn((1, 3, 512, 800), dtype=torch.float32)
    image=cv2.imread("dog.jpg")
    image=cv2.resize(image,(512,512))
    # image=Image.open("dog.jpg")
    transform=transforms.ToTensor()
    x=transform(image).unsqueeze(0)
    x=x.float()
    print(x.type())
    print(x.shape)
    # x=cv2.normalize(x, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # x=cv2.resize(x,(512,512))
    C2, C3, C4, C5 = net(x)

    ### torchvision version ###
    #net_tv = models.resnet18(pretrained=False)
    #net_tv = models.resnet34(pretrained=False)
    net_tv = models.resnet50(pretrained=False)
    #net_tv = models.resnet101(pretrained=False)
    #net_tv = models.resnet152(pretrained=False)
    C2_tv, C3_tv, C4_tv, C5_tv = GetFeatureMapsFromResnet(net_tv, x)

    print("Verifying the feature map shapes of customed ResNet and ResNet from torchvision")
    # print(f"C2.shape of customed ResNet: {C2.shape}")
    # print(C2.type())
    # # tensor_array_C2 = C2.cpu()
    # numpy_array_C2 = C2.detach().numpy()
    # print(numpy_array_C2.shape)
    # combined_image_C2 = numpy_array_C2[0].sum(axis=0)
    # plt.imsave('combined_image_C2.png', combined_image_C2, cmap='viridis', format='png')

    print(f"C2.shape of torchvision ResNet: {C2_tv.shape}")
    numpy_array_C2_tv = C2_tv.detach().numpy()
    print(numpy_array_C2_tv.shape)
    combined_image_C2_tv = numpy_array_C2_tv[0].sum(axis=0)
    plt.imsave('combined_image_C2_tv.png', combined_image_C2_tv, cmap='viridis', format='png')
    
    # print(f"C3.shape of customed ResNet: {C3.shape}")
    # print(C3.type())
    # # tensor_array_C3 = C3.cpu()
    # numpy_array_C3 = C3.detach().numpy()
    # print(numpy_array_C3.shape)
    # combined_image_C3 = numpy_array_C3[0].sum(axis=0)
    # plt.imsave('combined_image_C3.png', combined_image_C3, cmap='viridis', format='png')

    print(f"C3.shape of torchvision ResNet: {C3_tv.shape}")
    numpy_array_C3_tv = C3_tv.detach().numpy()
    print(numpy_array_C3_tv.shape)
    combined_image_C3_tv = numpy_array_C3_tv[0].sum(axis=0)
    plt.imsave('combined_image_C3_tv.png', combined_image_C3_tv, cmap='viridis', format='png')
    
    # print(f"C4.shape of customed ResNet: {C4.shape}")
    # print(C4.type())
    # # tensor_array_C4 = C4.cpu()
    # numpy_array_C4 = C4.detach().numpy()
    # print(numpy_array_C4.shape)
    # combined_image_C4 = numpy_array_C4[0].sum(axis=0)
    # plt.imsave('combined_image_C4.png', combined_image_C4, cmap='viridis', format='png')

    print(f"C4.shape of torchvision ResNet: {C4_tv.shape}")
    numpy_array_C4_tv = C4_tv.detach().numpy()
    print(numpy_array_C4_tv.shape)
    combined_image_C4_tv = numpy_array_C4_tv[0].sum(axis=0)
    plt.imsave('combined_image_C4_tv.png', combined_image_C4_tv, cmap='viridis', format='png')
    
    # print(f"C5.shape of customed ResNet: {C5.shape}")
    # print(C5.type())
    # # tensor_array_C5 = C5.cpu()
    # numpy_array_C5 = C5.detach().numpy()
    # print(numpy_array_C5.shape)
    # combined_image_C5 = numpy_array_C5[0].sum(axis=0)
    # plt.imsave('combined_image_C5.png', combined_image_C5, cmap='viridis', format='png')

    print(f"C5.shape of torchvision ResNet: {C5_tv.shape}")
    numpy_array_C5_tv = C5_tv.detach().numpy()
    print(numpy_array_C5_tv.shape)
    combined_image_C5_tv = numpy_array_C5_tv[0].sum(axis=0)
    plt.imsave('combined_image_C5_tv.png', combined_image_C5_tv, cmap='viridis', format='png')
    
    print("Done!")