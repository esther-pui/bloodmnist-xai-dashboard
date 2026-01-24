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
import os

# -----------------------------
# 1. Dynamic Metadata
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_flag = 'dermamnist'
info = INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])
label_names = [info['label'][str(i)] for i in range(len(info['label']))]
SIZE = 224

# -----------------------------
# 2. Model Architecture (UPDATED TO RESNET50)
# -----------------------------
class Bottleneck(nn.Module):
    """Bottleneck block for ResNet50"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
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
    def __init__(self, block, layers, in_channels=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights (Optional for inference, but good practice)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Added MaxPool (Critical for ResNet50)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet50(in_channels=3, num_classes=8):
    # ResNet50 Structure: [3, 4, 6, 3]
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

class PCAClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# -----------------------------
# 3. Initialization & Loading
# -----------------------------
# Load Main Model (ResNet50)
model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)

# Point this to her saved file (even if she named it resnet18, it is resnet50 structure)
model_path = "assets/dermamnist_resnet18.pth" 

try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Main Model (ResNet50) Loaded")
    else:
        print(f"⚠️ Model file not found at {model_path}")
except RuntimeError as e:
    print(f"❌ Error loading model: {e}")
    print("Tip: Ensure the .pth file was trained with the ResNet50 architecture.")
    
model.eval()

# Load PCA
try:
    pca = joblib.load("assets/pca_model.pkl")
    print("✅ PCA Loaded")
except:
    print("⚠️ PCA not found")

# Load PCA Classifier
pca_classifier = None
if pca:
    pca_classifier = PCAClassifier(pca.n_components, n_classes).to(device)
    try:
        pca_classifier.load_state_dict(torch.load("assets/pca_classifier.pth", map_location=device))
        print("✅ PCA Classifier Loaded")
    except:
        print("⚠️ PCA Classifier not found")
    pca_classifier.eval()

# DiCE Wrapper
class DiCEWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
    def forward(self, x):
        return self.classifier(x.to(device)).cpu()

wrapped_model = DiCEWrapper(pca_classifier)

# -----------------------------
# 4. Helpers
# -----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def get_gradcam_image(input_tensor, model):
    # ResNet50 usually targets layer4[-1] (the last bottleneck block)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    img_np = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

def get_counterfactuals_single(image, target_classes=None):

    # ---------------------------------------------------------
    # PART 1: PREPARATION
    # ---------------------------------------------------------
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)
    
    # Manual feature extraction must match ResNet50 forward pass exactly
    with torch.no_grad():
        x = F.relu(model.bn1(model.conv1(input_tensor)))
        x = model.maxpool(x)  # <--- Don't forget this!
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        # ResNet50 layer4 output is 2048 channels (512 * expansion 4)
        features = x.view(x.size(0), -1).cpu().numpy()

    if pca is None or pca_classifier is None:
        return None

    # PCA Transform
    embedding_pca = pca.transform(features)
    feat_cols = [f"PC{i}" for i in range(pca.n_components)]
    query_df = pd.DataFrame(embedding_pca, columns=feat_cols)
    
    with torch.no_grad():
        logits = pca_classifier(torch.tensor(embedding_pca).float().to(device))
        current_pred = torch.argmax(logits, dim=1).item()
    
    query_df['label'] = [current_pred]

    # ---------------------------------------------------------
    # PART 2: GENETIC SEARCH
    # ---------------------------------------------------------
    min_row = {col: -100.0 for col in feat_cols}
    max_row = {col: 100.0 for col in feat_cols}
    min_row['label'] = 0
    max_row['label'] = n_classes - 1
    bounds_df = pd.DataFrame([min_row, max_row])
    working_df = pd.concat([query_df, bounds_df], ignore_index=True)

    d = dice_ml.Data(dataframe=working_df, continuous_features=feat_cols, outcome_name='label')
    m = dice_ml.Model(model=wrapped_model, backend="PYT")
    exp = dice_ml.Dice(d, m, method="genetic")

    final_cf_df = None
    
    # probs = torch.nn.functional.softmax(logits, dim=1)
    # top_idxs = torch.topk(probs, k=3).indices[0].tolist()
    # target_candidates = [idx for idx in top_idxs if idx != current_pred]

    # If caller specifies target classes (e.g., benign only), use those
    if target_classes is not None:
        target_candidates = [idx for idx in target_classes if idx != current_pred]
    else:
        # Default behavior: nearest alternative classes
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_idxs = torch.topk(probs, k=3).indices[0].tolist()
        target_candidates = [idx for idx in top_idxs if idx != current_pred]


    print(f"DEBUG: Starting DiCE for Pred {current_pred}...")

    for target_class in target_candidates:
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instances=query_df.drop(columns=['label']), 
                total_CFs=1, 
                desired_class=int(target_class),
                proximity_weight=0.1, diversity_weight=0.5,
                sample_size=50, max_iterations=10 
            )
            temp_df = dice_exp.cf_examples_list[0].final_cfs_df.copy()
            if not temp_df.empty and int(temp_df.iloc[0]['label']) != current_pred:
                final_cf_df = temp_df
                break 
        except:
            continue

    # ---------------------------------------------------------
    # PART 3: BRUTE FORCE FALLBACK
    # ---------------------------------------------------------
    if final_cf_df is None:
        print("   > Genetic failed. Switching to Brute Force...")
        for i in range(1000):
            noise_level = 1.0 + (i * 0.5) 
            noise = np.random.normal(0, noise_level, size=embedding_pca.shape)
            potential_cf = embedding_pca + noise
            
            with torch.no_grad():
                p_logits = pca_classifier(torch.tensor(potential_cf).float().to(device))
                p_pred = torch.argmax(p_logits, dim=1).item()
            
            # if p_pred != current_pred:
            if p_pred != current_pred and (target_classes is None or p_pred in target_classes):

                final_cf_df = pd.DataFrame(potential_cf, columns=feat_cols)
                final_cf_df['label'] = [p_pred]
                break

    # ---------------------------------------------------------
    # PART 4: MAGNITUDE
    # ---------------------------------------------------------
    if final_cf_df is not None:
        query_vals = query_df.drop(columns=['label']).values
        cf_vals = final_cf_df.drop(columns=['label']).values
        delta = cf_vals - query_vals
        final_cf_df['change_magnitude'] = np.linalg.norm(delta, axis=1)
        return final_cf_df
    
    return None