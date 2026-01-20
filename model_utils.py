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
# 1. Dynamic Metadata (Matches Colab)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_flag = 'dermamnist'
info = INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])
label_names = [info['label'][str(i)] for i in range(len(info['label']))]
SIZE = 224

# -----------------------------
# 2. Model Architecture (Must match Teammate's code exactly)
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
# Load Main Model
model = ResNet18(in_channels=3, num_classes=n_classes).to(device)
try:
    model.load_state_dict(torch.load("assets/dermamnist_resnet18.pth", map_location=device))
    print("✅ Main Model Loaded")
except:
    print("⚠️ Main Model not found in assets/")
model.eval()

# Load PCA
try:
    pca = joblib.load("assets/pca_model.pkl")
    print("✅ PCA Loaded")
except:
    print("⚠️ PCA not found")

# Load PCA Classifier (for DiCE)
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
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    img_np = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

def get_counterfactuals_single(image):
    # ---------------------------------------------------------
    # PART 1: PREPARATION (Same as before)
    # ---------------------------------------------------------
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        x = F.relu(model.bn1(model.conv1(input_tensor)))
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        features_512 = x.view(x.size(0), -1).cpu().numpy()

    embedding_pca = pca.transform(features_512)
    feat_cols = [f"PC{i}" for i in range(pca.n_components)]
    query_df = pd.DataFrame(embedding_pca, columns=feat_cols)
    
    with torch.no_grad():
        logits = pca_classifier(torch.tensor(embedding_pca).float().to(device))
        current_pred = torch.argmax(logits, dim=1).item()
    
    query_df['label'] = [current_pred]

    # ---------------------------------------------------------
    # PART 2: ATTEMPT SMART GENETIC SEARCH
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
    
    # Try the top 2 alternatives
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
                sample_size=50, max_iterations=10 # Very fast check
            )
            temp_df = dice_exp.cf_examples_list[0].final_cfs_df.copy()
            if not temp_df.empty and int(temp_df.iloc[0]['label']) != current_pred:
                final_cf_df = temp_df
                print("   > Genetic Search Success!")
                break 
        except:
            continue

    # ---------------------------------------------------------
    # PART 3: THE "FAIL-SAFE" (Random Perturbation)
    # If Genetic Algorithm failed, use Brute Force to GUARANTEE a result
    # ---------------------------------------------------------
    if final_cf_df is None:
        print("   > Genetic failed. Switching to Brute Force Fallback...")
        
        # We try 1000 times with increasing noise until the label flips
        for i in range(1000):
            # Create random noise that gets stronger every loop
            noise_level = 1.0 + (i * 0.5) 
            noise = np.random.normal(0, noise_level, size=embedding_pca.shape)
            
            potential_cf = embedding_pca + noise
            
            # Check if this new "noisy" features flip the class
            with torch.no_grad():
                p_logits = pca_classifier(torch.tensor(potential_cf).float().to(device))
                p_pred = torch.argmax(p_logits, dim=1).item()
            
            if p_pred != current_pred:
                print(f"   > Brute Force Success at iter {i}! Flipped to {p_pred}")
                
                # Construct the DataFrame manually
                final_cf_df = pd.DataFrame(potential_cf, columns=feat_cols)
                final_cf_df['label'] = [p_pred]
                break

    # ---------------------------------------------------------
    # PART 4: CALCULATE MAGNITUDE
    # ---------------------------------------------------------
    if final_cf_df is not None:
        query_vals = query_df.drop(columns=['label']).values
        cf_vals = final_cf_df.drop(columns=['label']).values
        delta = cf_vals - query_vals
        final_cf_df['change_magnitude'] = np.linalg.norm(delta, axis=1)
        return final_cf_df
    
    return None # Should almost never happen now