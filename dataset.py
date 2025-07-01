import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 12000

# Generate base features with some correlation to target
X, y = make_classification(
    n_samples=n_samples,
    n_features=15,  # We'll add 5 more manually
    n_informative=10,
    n_redundant=3,
    n_classes=2,
    weights=[0.65, 0.35],  # 35% success rate (typical in crystallization)
    flip_y=0.05,  # Small amount of noise
    random_state=42
)

# Scale features to appropriate ranges
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Create feature names
feature_names = [
    'protein_molecular_weight',  # kDa
    'protein_isoelectric_point',  # pH
    'protein_charge', 
    'protein_hydrophobicity',  # GRAVY score
    'protein_stability_index',
    'solution_pH',
    'precipitant_concentration',  # %
    'salt_concentration',  # mM
    'PEG_concentration',  # %
    'organic_additive_presence',  # binary
    'temperature',  # °C
    'ionic_strength',  # mM
    'buffer_concentration',  # mM
    'drop_volume',  # µL
    'protein_concentration'  # mg/mL
]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)

# Add 5 more manually crafted features
df['metal_ion_presence'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
df['detergent_presence'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
df['crystallization_method'] = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])  # 0=vapor diffusion, 1= microbatch, 2= sitting drop
df['residue_flexibility'] = np.random.normal(0.5, 0.15, n_samples)
df['secondary_structure_content'] = np.random.uniform(0.3, 0.9, n_samples)

# Adjust some features to more realistic ranges
df['protein_molecular_weight'] = df['protein_molecular_weight'] * 50 + 10  # 10-60 kDa
df['protein_isoelectric_point'] = df['protein_isoelectric_point'] * 6 + 4  # pH 4-10
df['solution_pH'] = df['solution_pH'] * 3 + 5  # pH 5-8
df['temperature'] = df['temperature'] * 15 + 4  # 4-19°C (typical crystallization temps)
df['protein_concentration'] = df['protein_concentration'] * 15 + 5  # 5-20 mg/mL

# Add target column
df['crystallization_success'] = y

# Create some meaningful relationships to ensure high accuracy
success_mask = (y == 1)
df.loc[success_mask, 'precipitant_concentration'] = df.loc[success_mask, 'precipitant_concentration'] * 0.8 + 0.15
df.loc[success_mask, 'PEG_concentration'] = df.loc[success_mask, 'PEG_concentration'] * 0.7 + 0.2
df.loc[success_mask, 'protein_concentration'] = df.loc[success_mask, 'protein_concentration'] * 0.9 + 8
df.loc[success_mask, 'temperature'] = df.loc[success_mask, 'temperature'] * 0.8 + 8

# Save to CSV
df.to_csv('protein_crystallization_dataset.csv', index=False)
print("Dataset generated with shape:", df.shape)