import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MolecularImageModel(nn.Module):
    def __init__(self, channels, hidden_dim, latent_dim, dropout, use_layernorm, pool_type, attention_type, device):
        super(MolecularImageModel, self).__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.pool_type = pool_type
        self.attention_type = attention_type
        self.device = device

        # Convolutional layers
        self.conv_layers = []
        in_channels = 3
        for out_channels in self.channels:
            self.conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(device),
                nn.BatchNorm2d(out_channels).to(device),
                nn.ReLU().to(device),
                nn.MaxPool2d(2).to(device),
                nn.Dropout2d(self.dropout).to(device)
            ])
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # Functional groups attention
        self.functional_group_attention = FunctionalGroupAttention(
            self.channels[-1]
        ).to(device)

        # Molecularly characterized attention
        if self.attention_type != 'none':
            self.attention = MolecularAttention(
                self.channels[-1],
                attention_type=self.attention_type
            ).to(device)
        else:
            self.attention = nn.Identity().to(device)

        # Pooling
        if self.pool_type == 'adaptive_avg':
            self.pooling = nn.AdaptiveAvgPool2d(1).to(device)
        elif self.pool_type == 'adaptive_max':
            self.pooling = nn.AdaptiveMaxPool2d(1).to(device)
        elif self.pool_type == 'attention':
            self.pooling = AttentionPooling(self.channels[-1]).to(device)

        # Feature mapping
        self.fc1 = nn.Linear(self.channels[-1], self.hidden_dim).to(device)
        self.dropout1 = nn.Dropout(self.dropout).to(device)
        if self.use_layernorm:
            self.ln1 = nn.LayerNorm(self.hidden_dim).to(device)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device)
        self.dropout2 = nn.Dropout(self.dropout).to(device)

        self.fc3 = nn.Linear(self.hidden_dim, self.latent_dim).to(device)
        if self.use_layernorm:
            self.ln2 = nn.LayerNorm(self.latent_dim).to(device)

    def forward(self, x):

        features = self.conv_layers(x)

        features, functional_groups = self.functional_group_attention(features)

        features = self.attention(features)

        pooled = self.pooling(features)

        x = self.fc1(pooled.flatten(1))
        x = F.relu(x)
        x = self.dropout1(x)
        if self.use_layernorm:
            x = self.ln1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if self.use_layernorm:
            x = self.ln2(x)

        return x, functional_groups


class FunctionalGroupAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Chemical bonding parameters (in Å)
        self.bond_lengths = {
            'C-H': 1.09,
            'C-C': 1.54,
            'C=C': 1.34,
            'C≡C': 1.20,
            'C-O': 1.43,
            'C=O': 1.23,
            'O-H': 0.96,
            'N-H': 1.01,
            'C-N': 1.47,
            'C=N': 1.28,
            'C≡N': 1.16,
            'C-S': 1.82,
            'S-H': 1.34,
            'P-O': 1.63,
            'P=O': 1.50
        }

        # Key angle parameters (in degrees)
        self.bond_angles = {
            'C-C-C': 109.5,
            'C=C-C': 120,
            'C-O-H': 109,
            'O=C-O': 123,
            'C-N-H': 109,
            'O=C-N': 120,
            'C-S-H': 96,
            'O=S=O': 119,
            'O=P-O': 117
        }

        # Electronegativity values
        self.electronegativity = {
            'H': 2.20,
            'C': 2.55,
            'N': 3.04,
            'O': 3.44,
            'F': 3.98,
            'Cl': 3.16,
            'Br': 2.96,
            'I': 2.66,
            'S': 2.58,
            'P': 2.19
        }

        # Define functional group templates and their chemical properties
        self.functional_group_patterns = {
            'hydroxyl': {  # -OH
                'composition': ['C', 'O', 'H'],
                'bonds': ['C-O', 'O-H'],
                'angles': ['C-O-H']
            },
            'carboxyl': {  # -COOH
                'composition': ['C', 'O', 'O', 'H'],
                'bonds': ['C=O', 'C-O', 'O-H'],
                'angles': ['O=C-O']
            },
            'carbonyl': {  # C=O
                'composition': ['C', 'O'],
                'bonds': ['C=O'],
                'angles': []
            },
            'amino': {  # -NH2
                'composition': ['C', 'N', 'H', 'H'],
                'bonds': ['C-N', 'N-H', 'N-H'],
                'angles': ['C-N-H']
            },
            'amide': {  # -CONH2
                'composition': ['C', 'O', 'N', 'H', 'H'],
                'bonds': ['C=O', 'C-N', 'N-H', 'N-H'],
                'angles': ['O=C-N']
            },
            'ether': {  # R-O-R
                'composition': ['C', 'O', 'C'],
                'bonds': ['C-O', 'C-O'],
                'angles': ['C-O-C']
            },
            'ester': {  # R-COO-R
                'composition': ['C', 'O', 'O', 'C'],
                'bonds': ['C=O', 'C-O'],
                'angles': ['O=C-O']
            },
            'alkene': {  # C=C
                'composition': ['C', 'C'],
                'bonds': ['C=C'],
                'angles': []
            },
            'alkyne': {  # C≡C
                'composition': ['C', 'C'],
                'bonds': ['C≡C'],
                'angles': []
            },
            'thiol': {  # -SH
                'composition': ['C', 'S', 'H'],
                'bonds': ['C-S', 'S-H'],
                'angles': ['C-S-H']
            },
            'sulfoxide': {  # R-SO-R
                'composition': ['C', 'S', 'O', 'C'],
                'bonds': ['C-S', 'S=O', 'C-S'],
                'angles': ['O=S-C']
            },
            'sulfone': {  # R-SO2-R
                'composition': ['C', 'S', 'O', 'O', 'C'],
                'bonds': ['C-S', 'S=O', 'S=O', 'C-S'],
                'angles': ['O=S=O']
            },
            'phosphate': {  # -PO4
                'composition': ['P', 'O', 'O', 'O', 'O'],
                'bonds': ['P=O', 'P-O', 'P-O', 'P-O'],
                'angles': ['O=P-O']
            },
            'nitrile': {  # -C≡N
                'composition': ['C', 'N'],
                'bonds': ['C≡N'],
                'angles': []
            },
            'nitro': {  # -NO2
                'composition': ['N', 'O', 'O'],
                'bonds': ['N=O', 'N=O'],
                'angles': ['O=N=O']
            }
        }

        # Define functional group templates and their chemical properties
        for group_name, properties in self.functional_group_patterns.items():
            properties['pattern'] = self.generate_template(
                bonds=properties['bonds'],
                central_atom=properties['composition'][0],
                angle=properties['angles'][0] if properties['angles'] else None
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.initialize_templates(in_channels)

        self.attention = nn.Sequential(
            nn.Conv2d(len(self.functional_group_patterns), 1, 1),
            nn.Sigmoid()
        )

    def generate_template(self, bonds, central_atom, angle=None, resolution=3):
        """Generate templates based on chemical parameters"""
        template = np.zeros((resolution, resolution))
        center_x, center_y = resolution // 2, resolution // 2

        # Electronegativity contribution of the central atom
        template[center_x, center_y] = self.electronegativity[central_atom] / 4.0

        # Keyholder contributions
        for bond in bonds:
            if bond in self.bond_lengths:
                bond_length = self.bond_lengths[bond]
                normalized_length = (bond_length - 0.9) / (1.5 - 0.9)

                if angle and angle in self.bond_angles:
                    angle_rad = math.radians(self.bond_angles[angle])
                    dx = normalized_length * math.cos(angle_rad)
                    dy = normalized_length * math.sin(angle_rad)

                    x = int(center_x + dx)
                    y = int(center_y + dy)
                    if 0 <= x < resolution and 0 <= y < resolution:
                        template[x, y] = normalized_length
                else:
                    template[center_x, min(center_y + 1, resolution-1)] = normalized_length

        return template

    def initialize_templates(self, in_channels):
        """Initialize functional group templates"""
        templates = []
        for group_info in self.functional_group_patterns.values():
            pattern = group_info['pattern']
            pattern_tensor = torch.tensor(pattern, dtype=torch.float32)
            template = pattern_tensor.unsqueeze(0).repeat(in_channels, 1, 1)
            templates.append(template)

        self.templates = nn.Parameter(
            torch.stack(templates)
        )

    def forward(self, x):
        features = self.conv(x)
        template_matches = []

        for template in self.templates:
            match = F.conv2d(features, template.unsqueeze(0), padding=1)
            template_matches.append(match)

        matches = torch.cat(template_matches, dim=1)
        attention_weights = self.attention(matches)

        return x * attention_weights.expand_as(x), matches


class MolecularAttention(nn.Module):
    def __init__(self, channels, attention_type='both'):
        super().__init__()
        self.attention_type = attention_type

        if attention_type in ['spatial', 'both']:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Sigmoid()
            )

        if attention_type in ['channel', 'both']:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channels // 16, channels, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.attention_type in ['spatial', 'both']:
            spatial_weights = self.spatial_attention(x)
            x = x * spatial_weights

        if self.attention_type in ['channel', 'both']:
            channel_weights = self.channel_attention(x)
            x = x * channel_weights

        return x


class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return (x * weights).sum(dim=(2, 3), keepdim=True)


