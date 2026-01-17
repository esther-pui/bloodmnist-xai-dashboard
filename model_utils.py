# model_utils.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import joblib
import dice_ml

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# BloodMNIST parameters
# -----------------------------
n_channels = 1
label_names = [
    'Neutrophil', 'Eosinophil', 'Basophil', 'Lymphocyte', 
    'Monocyte', 'Immature Granulocyte', 'Red Blood Cell', 'Platelet'
]
n_classes = len(label_names)

# -----------------------------
# ResNet18 Model Definition
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=8):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(in_channels=1, num_classes=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)

# -----------------------------
# Load pretrained ResNet18 model
# -----------------------------
model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
model.load_state_dict(torch.load("bloodmnist_resnet18.pth", map_location=device))
model.eval()

# -----------------------------
# PCA + PCAClassifier for DiCE
# -----------------------------
from sklearn.decomposition import PCA

class PCAClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Load PCA and PCA classifier
pca = joblib.load("pca_model.pkl")
pca_classifier = PCAClassifier(pca.n_components, n_classes).to(device)
pca_classifier.load_state_dict(torch.load("pca_classifier.pth", map_location=device))
pca_classifier.eval()

# Wrap PCA classifier for DiCE
class DiCEWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        return self.classifier(x).cpu()

wrapped_model = DiCEWrapper(pca_classifier).to(device)

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*n_channels, std=[0.5]*n_channels)
    ])
    return transform(image)

# -----------------------------
# Grad-CAM visualization
# -----------------------------
def get_gradcam_image(input_tensor, model):
    model.eval()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    img_np = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return cam_image

# -----------------------------
# DiCE counterfactuals
# -----------------------------
def get_counterfactuals_single(input_tensor, pca, pca_classifier, wrapped_model):
    # Placeholder embedding (extract real embeddings for full dashboard)
    with torch.no_grad():
        embedding_pca = np.random.randn(1, pca.n_components)

    df = pd.DataFrame(embedding_pca)
    df['label'] = [0]  # placeholder

    continuous_features = df.columns[:-1].tolist()
    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='label')
    m = dice_ml.Model(model=wrapped_model, backend="PYT")
    exp = dice_ml.Dice(d, m, method="genetic")

    query_instance = df.drop(columns=['label']).iloc[0:1]
    cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class=1)

    cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
    cf_df['change_magnitude'] = cf_df.drop(columns=['label']).sub(query_instance.values[0]).abs().sum(axis=1)
    cf_df = cf_df.sort_values('change_magnitude')
    cf_transposed = cf_df.drop(columns=['change_magnitude']).transpose()
    cf_transposed.columns = [f'CF{i+1}' for i in range(cf_transposed.shape[1])]

    return cf_df, cf_transposed
