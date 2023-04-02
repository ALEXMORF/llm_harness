import torch
import math

class SelfAttention(torch.nn.Module):
    def __init__(self, input_size, head_size, value_size):
        super().__init__()
        self.head_size_sqrt = math.sqrt(head_size)
        self.input2k = torch.nn.Linear(input_size, head_size, bias=False)
        self.input2q = torch.nn.Linear(input_size, head_size, bias=False)
        self.input2v = torch.nn.Linear(input_size, value_size, bias=False)

    def forward(self, xs):
        Q = self.input2q(xs) # B, C, H
        K = self.input2k(xs) # B, C, H
        V = self.input2v(xs) # B, C, V
        w = torch.bmm(Q, torch.transpose(K, -1, -2)) # B, C, C
        mask = torch.tril(torch.ones(w.shape[1:])) # C, C
        w = w.masked_fill(mask == 0, float('-inf'))
        w = torch.nn.functional.softmax(w / self.head_size_sqrt, dim=-1)
        return torch.bmm(w, V) # B, C, V