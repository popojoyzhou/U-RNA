import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import time
import random
import joblib
import pickle
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import time
import random
import joblib
import torch.optim as optim
# Optimizer
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # Additional layers
        self.output_conv = nn.AdaptiveAvgPool1d(64)
        self.fc = nn.Linear(64, 4)
        self.conv1_structure = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv1_sub1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv1_sub2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)

    def pad_if_needed(self, x):
        h, w = x.size(2), x.size(3)
        pad_h = (h % 2 != 0)
        pad_w = (w % 2 != 0)
        if pad_h or pad_w:
            x = F.pad(x, (0, int(pad_w), 0, int(pad_h)))
        return x

    def center_crop(self, layer, target_height, target_width):
        _, _, h, w = layer.size()
        diff_y = (h - target_height) // 2
        diff_x = (w - target_width) // 2
        return layer[:, :, diff_y:(diff_y + target_height), diff_x:(diff_x + target_width)]

    def forward(self, x):
        x_structure, x_sub1, x_sub2 = x
        x_structure = x_structure.permute(0, 3, 1, 2)  # Expected shape: [batch_size, 1, 512, 512]
        x_sub1 = x_sub1.permute(0, 3, 1, 2)  # Expected shape: [batch_size, 14, 512, 512]
        x_sub2 = x_sub2.permute(0, 3, 1, 2)  # Expected shape: [batch_size, 4, 512, 512]

        # Initial convolution layers
        x1 = torch.relu(self.conv1_structure(x_structure))
        x2 = torch.relu(self.conv1_sub1(x_sub1))
        x3 = torch.relu(self.conv1_sub2(x_sub2))
        x = torch.cat([x1, x2, x3], dim=1)  # Shape: [batch_size, 64, 512, 512]

        # Downsampling path with padding
        e1 = self.Conv1(x)

        e2 = self.pad_if_needed(e1)
        e2 = self.Maxpool1(e2)
        e2 = self.Conv2(e2)

        e3 = self.pad_if_needed(e2)
        e3 = self.Maxpool2(e3)
        e3 = self.Conv3(e3)

        e4 = self.pad_if_needed(e3)
        e4 = self.Maxpool3(e4)
        e4 = self.Conv4(e4)

        e5 = self.pad_if_needed(e4)
        e5 = self.Maxpool4(e5)
        e5 = self.Conv5(e5)
        # Upsampling + Cropping + Attention + Concatenation
        d5 = self.Up5(e5)
        d5 = self.center_crop(d5, e4.size(2), e4.size(3))  # Crop d5 to match e4
        x4 = self.Att5(g=d5, x=e4)
        
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.center_crop(d4, e3.size(2), e3.size(3))  # Crop d4 to match e3
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = self.center_crop(d3, e2.size(2), e2.size(3))  # Crop d3 to match e2
        x2 = self.Att3(g=d3, x=e2)
        
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.center_crop(d2, e1.size(2), e1.size(3))  # Crop d2 to match e1
        x1 = self.Att2(g=d2, x=e1)
        
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = out.permute(0, 2, 3, 1)
        #print(out.shape)
        out = out.reshape(out.size(0), out.size(1), out.size(2) * 16)
        x = self.output_conv(out)
        x = self.fc(x)
        #x = out.permute(0, 2, 1)
        #out = out.reshape(out.size(0), out.size(1), out.size(2) * 16)
        #x = self.fc(out)

        return x 


def one_hot_to_sequence(one_hot_tensor):
    """Convert one-hot encoded tensor to RNA sequence string."""
    # Find the index with the maximum probability along the last dimension
    indices = torch.argmax(one_hot_tensor, dim=-1)  # Shape: [batch, len]

    # Map indices to nucleotides
    nucleotide_map = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
    batch_sequences = []
    for i in range(indices.size(0)):  # Iterate over the batch
        sequence = ''.join([nucleotide_map[idx.item()] for idx in indices[i]])
        batch_sequences.append(sequence)
    
    return batch_sequences
class RNADesignData(Dataset):
    def __init__(self, encoded_bp, encoded_seq, encoded_dotb, encoded_sub1, encoded_sub2):
        self.encoded_bp = encoded_bp
        self.encoded_seq = encoded_seq
        self.encoded_dotb = encoded_dotb
        self.encoded_sub1 = encoded_sub1
        self.encoded_sub2 = encoded_sub2

    def __len__(self):
        return len(self.encoded_seq)
    
    def stacking(self, x):
        seq_len = x.shape[0]  # Dynamically get the sequence length (or first dimension size)

        # Broadcast the tensor along the new axis
        first_broadcast = np.tile(x[:, np.newaxis, :], (1, seq_len, 1))

        # Expand encoded_sub1 to [seq_len, seq_len, channels] by repeating along the first dimension
        second_broadcast = np.tile(x[np.newaxis, :, :], (seq_len, 1, 1))

        # Stack them along the last dimension to get a [seq_len, seq_len, 2 * channels] tensor
        stacked_tensor = np.concatenate((first_broadcast, second_broadcast), axis=-1)

        return stacked_tensor

    def __getitem__(self, idx):
        bp = torch.tensor(self.encoded_bp[idx], dtype=torch.float32)
        bp = torch.unsqueeze(bp,2)
        seq = torch.tensor(self.encoded_seq[idx], dtype=torch.int64)
        sub1 = torch.tensor(self.stacking(self.encoded_sub1[idx]), dtype=torch.float32)
        sub2 = torch.tensor(self.stacking(self.encoded_sub2[idx]), dtype=torch.float32)
        
        inputs = (bp, sub1, sub2)
        
        return inputs, seq


import torch
import torch.nn as nn
   

def masked_cross_entropy_loss(output, target, ss_struct):
    # Transpose the target to have the same dimension order as output
    target = target.permute(0, 2, 1)  # Shape: [batch_size, 512, 4]
    target_class = torch.argmax(target, dim=-1)  # Shape: [batch_size, 512]
    
    # Create a mask where the target has non-zero entries
    mask = torch.sum(target, dim=-1) != 0  # Shape: [batch_size, 512]

    # Calculate cross entropy loss
    loss = F.cross_entropy(output, target_class, reduction='none')  # Shape: [batch_size, 512]

    # Scaling based on base pair information in ss_struct
    # ss_struct is expected to be of shape [batch_size, len, len, 1]
    # We need to calculate a scaling factor matrix based on ss_struct
    ss_struct = ss_struct.squeeze(-1)  # Remove the last dimension -> [batch_size, len, len]
    
    # Create the scaling matrix: scale by 300 where there's a base pair, else 1
    scaling_matrix = torch.ones_like(ss_struct)  # Initialize with ones
    scaling_matrix[ss_struct != 0] = 300  # Scale by 300 where there's a base pair
    scaling_matrix = scaling_matrix.unsqueeze(-1)
    # Apply scaling to the loss
    loss = loss * mask * scaling_matrix

    # Normalize loss by the sum of the mask
    loss = loss.sum() / mask.sum()
    return loss

