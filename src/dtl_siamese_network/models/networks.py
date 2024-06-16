import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from dtl_siamese_network.data.transforms import get_test_transforms

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TorhModelFeatureExtraction(nn.Module):
    def __init__(self, name):
        super(TorhModelFeatureExtraction, self).__init__()
        if name == 'resnet50':
            self.model = models.resnet50(weights=True)
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256),
                                          nn.PReLU(),
                                          nn.Linear(256, 256),
                                          nn.PReLU(),
                                          nn.Linear(256, 128)
                                          )
        elif name == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
            self.model.classifier = nn.Sequential(nn.Linear(self.model.classifier[1].in_features, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128)
                                )
        else:
            raise NotImplementedError

        # self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        # self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 128)
        #                         )

    def forward(self, x):
        output = self.model(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(weights=True)
        # Изменение последнего слоя для получения 128-мерного вектора признаков
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def predict(self, img, device='cpu'):
        transform = get_test_transforms()
        transformed_image = transform(img)
        output = self.embedding_net(transformed_image.unsqueeze(0).to(device))
        return output

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
