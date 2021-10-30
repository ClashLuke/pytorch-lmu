import numpy as np

import torch
from torch import nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete


class LMUFFT(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta, fft_features):

        super(LMUFFT, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features=input_size, out_features=fft_features + hidden_size)
        self.W_h = nn.Linear(in_features=memory_size * fft_features, out_features=hidden_size)

        Q = np.arange(self.memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0) ** (i - j + 1))
        B = R * ((-1.0) ** Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system=(A, B, C, D), 
            dt=1.0, 
            method="zoh"
        )

        # To torch.tensor
        A = torch.from_numpy(A).float()  # [memory_size, memory_size]
        B = torch.from_numpy(B).float()  # [memory_size, 1]
        
        H = []
        A_i = torch.eye(self.memory_size)
        for t in range(self.seq_len):
            H.append(A_i @ B)
            A_i = A @ A_i

        H = torch.cat(H, dim=-1)  # [memory_size, seq_len]
        fft_H = fft.rfft(H, n=2*self.seq_len, dim=-1)  # [memory_size, seq_len + 1]

        self.register_buffer("fft_H", fft_H.unsqueeze(0))  # [1, memory_size, seq_len + 1]

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        out = self.W_u(x)
        out0 = out[:, :, :self.hidden_size]
        u = torch.relu(out[:, :, self.hidden_size:) # [batch_size, seq_len, 1]
        fft_input = u.transpose(1, 2) # [batch_size, 1, seq_len]
        fft_u = fft.rfft(fft_input, n=2 * seq_len, dim=-1) # [batch_size, seq_len, seq_len+1]
        temp = fft_u.view(-1, 1, seq_len) * self.fft_H  # [batch_size, memory_size, seq_len+1]
        m = fft.irfft(temp, n=2 * seq_len, dim=-1) # [batch_size, memory_size, seq_len+1]
        m = m[:, :, :-1] # [batch_size, memory_size, seq_len]
        m = m.view(batch_size, -1, seq_len)
        m = m.transpose(1, 2) # [batch_size, seq_len, memory_size]
        return torch.relu(self.W_h(m) + out0) # [batch_size, seq_len, hidden_size]
