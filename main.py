import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import h5py

# List of simple functional groups for classification (you can expand)
FUNCTIONAL_GROUPS = {
    'alcohol': 'O',  # Ethanol: OH
    'ketone': '[C;D3](=O)',  # keton
    'alkene': 'C=C',  # Alken
    'aromatic': 'c1ccccc1',  # aromatic
    'none': ''  # none
}

"""
    Extracting functional groups from SMILES with RDKit
    """
def extract_functional_groups(smiles: str) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ['none']

    groups = []
    for name, pattern in FUNCTIONAL_GROUPS.items():
        if pattern == '':
            continue
        substruct = Chem.MolFromSmarts(pattern)
        if mol.HasSubstructMatch(substruct):
            groups.append(name)

    return groups if groups else ['none']

# Preprocessing
"""
   Converting IR Spectrum(1D) to image(2D) for CNN. Hypothesize: IR Spectrum => wavenumber,intensity
    """
def spectrum_to_image(ir_spectrum: np.ndarray, img_size: tuple = (224, 224)) -> np.ndarray:

    wavenumbers = ir_spectrum[0]  #  4000-400 cm^-1
    intensities = ir_spectrum[1]

    # Normalize intensity to 0-1
    intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities) + 1e-8)

    # Resample to fixed length (e.g., 1024 points)
    fixed_len = 1024
    from scipy.interpolate import interp1d
    f = interp1d(wavenumbers, intensities, kind='linear', fill_value=0, bounds_error=False)
    x_new = np.linspace(wavenumbers.min(), wavenumbers.max(), fixed_len)
    intensities_resampled = f(x_new)

    # Convert to 2D image: height=1, width=fixed_len, channels=1
    image = intensities_resampled.reshape(1, fixed_len, 1).astype(np.float32)

    # Resize to img_size if needed (using simple repeat for demo)
    # For real, use torchvision.transforms.Resize
    return np.repeat(image, img_size[0], axis=0)[:, :img_size[1], :]  # Simple resize

# Loding dataset
"""
    Load dataset from HDF5 (based on multimodal dataset format).
Assumptions: keys like 'smiles', 'ir' (array of [wavenum, int]).
    """
def load_dataset(h5_path: str, num_samples: int = 10000) -> Dict:
    data = {'images': [], 'labels': [], 'smiles': []}
    with h5py.File(h5_path, 'r') as f:
        # It is assumed that the dataset is in /train/smiles and /train/ir (based on the GitHub repo)
        smiles_list = list(f['train/smiles'][:num_samples])  # Limit for demo
        ir_list = list(f['train/ir'][:num_samples])

        for smiles, ir in zip(smiles_list, ir_list):
            smiles_str = smiles.decode('utf-8') if isinstance(smiles, bytes) else str(smiles)
            groups = extract_functional_groups(smiles_str)
            label = np.eye(len(FUNCTIONAL_GROUPS))[list(FUNCTIONAL_GROUPS.keys()).index(groups[0])]  # One-hot for first group

            img = spectrum_to_image(ir)
            data['images'].append(img)
            data['labels'].append(label)
            data['smiles'].append(smiles_str)

    return {
        'images': np.array(data['images']),
        'labels': np.array(data['labels']),
        'smiles': data['smiles']
    }
# Creat model 1 with Pytorch    
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Simple CNN for classifying IR spectrum images.
Input: (batch, 1, 224, 224) - grayscale image from spectrum.
Output: num_classes (e.g., 5 functional groups).
    """
class IRCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(IRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # After pooling: 224/8=28
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Model testing
if __name__ == "__main__":
    model = IRCNN(num_classes=5)
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # torch.Size([1, 5])
    
    
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from model import IRCNN
from utils import load_dataset

# Train
# parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
H5_PATH = 'data/multimodal_spectroscopic_dataset.h5'  # After unzip, find the original HDF5 file.

 # Data loading
def train_model():
    data = load_dataset(H5_PATH)
    X = data['images']
    y = data['labels']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # To tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IRCNN(num_classes=y.shape[1]).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, torch.argmax(batch_y, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            true_labels.extend(torch.argmax(batch_y, dim=1).numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(true_labels, preds, target_names=list(FUNCTIONAL_GROUPS.keys())))

    # Save the model
    torch.save(model.state_dict(), 'ir_cnn_model.pth')
    return model

if __name__ == "__main__":
    model = train_model()
    
# predict for new molcular
import torch
from model import IRCNN
from utils import extract_functional_groups, spectrum_to_image
# Assumption: You have a new IR spectrum (np.array)

def predict_group(model_path: str, ir_spectrum: np.ndarray):
    model = IRCNN(num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img = spectrum_to_image(ir_spectrum)
    input_tensor = torch.tensor(img).unsqueeze(0)  # (1, 1, 224, 224)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        group_names = list(FUNCTIONAL_GROUPS.keys())
        predicted_group = group_names[pred_idx]

    return predicted_group

# Usage example
# dummy_spectrum = np.array([[np.linspace(400, 4000, 1000), np.random.rand(1000)]])  # [wavenum, int]
# print(predict_group('ir_cnn_model.pth', dummy_spectrum)) 

# Creating model with Method 2 using Tensorflow:

    import tensorflow as tf
from tensorflow.keras import layers, models

"""
    CNN model for classifying IR spectral images.
    """

#Input: (batch, 224, 224, 1) #-Gray image of the spectrum.
#Output: num_classes (5 functional groups).
def create_ircnn_model(num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Test model 2
if __name__ == "__main__":
    model = create_ircnn_model(num_classes=5)
    model.summary()  # Show model structure
# Output: something like 1.5M parameters and layers

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from model import create_ircnn_model
from utils import load_dataset, FUNCTIONAL_GROUPS


# Train
# parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
H5_PATH = 'data/multimodal_spectroscopic_dataset.h5'

def train_model():
    # Data loading
    data = load_dataset(H5_PATH, num_samples=10000)
    X = data['images']  # shape: (N, 224, 224, 1)
    y = data['labels']  # shape: (N, num_classes)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    model = create_ircnn_model(num_classes=len(FUNCTIONAL_GROUPS))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=(X_test, y_test),
                       verbose=1)

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true_classes, y_pred_classes,
                              target_names=list(FUNCTIONAL_GROUPS.keys())))

    #  saveing model
    model.save('ir_cnn_model.h5')
    return model

if __name__ == "__main__":
    model = train_model()

 #Prediction
import tensorflow as tf
import numpy as np
from utils import spectrum_to_image, FUNCTIONAL_GROUPS

def predict_group(model_path: str, ir_spectrum: np.ndarray):
    # Loadind data
    model = tf.keras.models.load_model(model_path)

    # Convert spectrum to image
    img = spectrum_to_image(ir_spectrum)  # shape: (224, 224, 1)
    input_tensor = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 1)

    # prediction
    pred = model.predict(input_tensor)
    pred_idx = np.argmax(pred, axis=1)[0]
    group_names = list(FUNCTIONAL_GROUPS.keys())
    predicted_group = group_names[pred_idx]

    return predicted_group

# Usage example
if __name__ == "__main__":
    # Assumption: Has IR spectrum
    dummy_spectrum = np.array([np.linspace(400, 4000, 1000), np.random.rand(1000)])
    print(predict_group('ir_cnn_model.h5', dummy_spectrum)) 
    
# Define madule Descriptor =>1) to obtain Molecular weight and Division factor(LogP)
from rdkit import Chem
from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles('CCO')  # Ethanol
mw = Descriptors.MolWt(mol)  # Molecular weight
logp = Descriptors.MolLogP(mol)  # Division factor (LogP)
print(f"Molecular weight: {mw}, LogP: {logp}")
# outpot: Molecular weight: 46.069, LogP: 0.0014

# 2) Hydrogen bond acceptor
from rdkit.Chem import rdMolDescriptors
mol = Chem.MolFromSmiles('CCO')  # Ethanol
hba = rdMolDescriptors.CalcNumHBA(mol)  # Number of Hydrogen bond acceptor
print(f"Hydrogen bond acceptor: {hba}")
# outpot: Hydrogen bond acceptor : 1

# To expanding the project
def extract_features(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    return {
        'mol_weight': Descriptors.MolWt(mol),
        'num_hba': rdMolDescriptors.CalcNumHBA(mol),
        'functional_groups': extract_functional_groups(smiles)