# model_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import joblib
import medmnist
from medmnist import INFO
import dice_ml

# -----------------------------
# 1. Dynamic Metadata (Colab Logic)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_flag = 'bloodmnist'
info = INFO[data_flag]
label_names = [info['label'][str(i)].replace('-', ' ').title() for i in range(len(info['label']))]
n_classes = len(label_names)
SIZE = 224 # From your Colab: SIZE = 224

# -----------------------------
# 2. Model Architecture (Exact Colab Classes)
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=8):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
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
        return self.linear(out)

def ResNet18(in_channels=3, num_classes=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)

class PCAClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# -----------------------------
# 3. Initialization & Loading
# -----------------------------
model = ResNet18(in_channels=3, num_classes=n_classes).to(device)
model.load_state_dict(torch.load("assets/bloodmnist_resnet18.pth", map_location=device))
model.eval()

pca = joblib.load("assets/pca_model.pkl")

pca_classifier = PCAClassifier(pca.n_components, n_classes).to(device)
pca_classifier.load_state_dict(torch.load("assets/pca_classifier.pth", map_location=device))
pca_classifier.eval()

class DiCEWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
    def forward(self, x):
        return self.classifier(x.to(device)).cpu()

wrapped_model = DiCEWrapper(pca_classifier)

# -----------------------------
# 4. Helpers (Colab Logic)
# -----------------------------
def preprocess_image(image):
    """Adjusted to 3 channels to match your saved .pth weights"""
    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.Lambda(lambda x: x.convert("RGB")), # Ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 3 values
    ])
    return transform(image)

def get_gradcam_image(input_tensor, model):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    img_np = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    # Note: show_cam_on_image expects RGB, it will handle the single channel
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

def get_counterfactuals_single(image):
    # 1. Feature Extraction
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        x = F.relu(model.bn1(model.conv1(input_tensor)))
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        features_512 = x.view(x.size(0), -1).cpu().numpy()

    # 2. PCA & Prediction
    embedding_pca = pca.transform(features_512)
    feat_cols = [f"PC{i}" for i in range(pca.n_components)]
    
    # 3. Create the Query DataFrame
    query_df = pd.DataFrame(embedding_pca, columns=feat_cols)
    
    with torch.no_grad():
        logits = pca_classifier(torch.tensor(embedding_pca).float().to(device))
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, 2)
        current_pred = top_idxs[0][0].item()
        target_class = top_idxs[0][1].item()

    query_df['label'] = [current_pred]

    # 4. CRITICAL FIX: Define a wide search range for DiCE
    # We create a dummy dataframe with min (-10) and max (+10) values for every PC
    # This tells DiCE it is allowed to search anywhere in this space.
    min_row = {col: -20.0 for col in feat_cols}
    max_row = {col: 20.0 for col in feat_cols}
    min_row['label'] = 0
    max_row['label'] = 1
    
    # Combine query with bounds to define the Data object
    bounds_df = pd.DataFrame([min_row, max_row])
    working_df = pd.concat([query_df, bounds_df], ignore_index=True)

    d = dice_ml.Data(dataframe=working_df, continuous_features=feat_cols, outcome_name='label')
    m = dice_ml.Model(model=wrapped_model, backend="PYT")
    exp = dice_ml.Dice(d, m, method="random")

    # 5. Generate Counterfactuals
    query_instance = query_df.drop(columns=['label'])
    
    try:
        dice_exp = exp.generate_counterfactuals(
            query_instance, 
            total_CFs=1, 
            desired_class=int(target_class),
            sample_size=200, 
            random_seed=42
        )
    except:
        # Fallback loop
        dice_exp = None
        for i in range(n_classes):
            if i == current_pred: continue
            try:
                dice_exp = exp.generate_counterfactuals(
                    query_instance, total_CFs=1, desired_class=int(i), sample_size=50
                )
                if dice_exp: break
            except:
                continue

    if dice_exp is None:
        return None

    cf_df = dice_exp.cf_examples_list[0].final_cfs_df.copy()
    
    if cf_df.empty or cf_df.iloc[0]['label'] == current_pred:
        return None

    delta = cf_df.drop(columns=['label']).values - query_instance.values
    cf_df['change_magnitude'] = np.linalg.norm(delta, axis=1)
    
    return cf_df