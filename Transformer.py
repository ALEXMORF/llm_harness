import torch
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoding(torch.nn.Module):
    def __init__(self, input_size, embed_size, context_len):
        super().__init__()
        self.token_embed = torch.nn.Embedding(input_size, embed_size)
        self.position_embed = torch.nn.Embedding(context_len, embed_size)
        self.context_len = context_len
    
    def forward(self, x):
        tok = self.token_embed(x) 
        pos = self.position_embed(torch.arange(0, x.shape[1]).to(device))
        return tok + pos

class MaskedSelfAttention(torch.nn.Module):
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
        mask = torch.tril(torch.ones(w.shape[1:])).to(device) # C, C
        w = w.masked_fill(mask == 0, float('-inf'))
        w = torch.nn.functional.softmax(w / self.head_size_sqrt, dim=-1)
        return torch.bmm(w, V) # B, C, V

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, head_size, head_count, value_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([MaskedSelfAttention(input_size, head_size, value_size) for _ in range(head_count)])
        self.combine = torch.nn.Linear(value_size * head_count, value_size)
    
    def forward(self, xs):
        out = torch.concat([head(xs) for head in self.heads], dim=-1)
        return self.combine(out)

class Block(torch.nn.Module):
    def __init__(self, input_size, head_size, head_count, value_size):
        super().__init__()
        self.attention = MultiHeadAttention(input_size, head_size, head_count, value_size)
        self.FC = torch.nn.Linear(value_size, input_size)
        self.layerNorm = torch.nn.LayerNorm(input_size)
    def forward(self, xs):
        out = self.attention(xs)
        out = self.layerNorm(self.FC(out) + xs)
        return out

def make_model(alphabet_size, context_len, embed_size, hidden_size, head_size=64, value_size=128, block_count=1, head_count=1):
    model = torch.nn.Sequential(
        Encoding(alphabet_size, embed_size, context_len),
        #MaskedSelfAttention(embed_size, head_size, value_size),
        *[Block(embed_size, head_size, head_count, value_size) for _ in range(block_count)],
        #Block(embed_size, head_size, value_size),
        #Block(embed_size, head_size, value_size),
        torch.nn.Linear(value_size, hidden_size), torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, alphabet_size)
    )
    return model

def test():
    batch_size = 3
    context_len = 4
    embed_size = 2
    x = torch.randn((batch_size, context_len, embed_size))
    print(x.shape)
    attention = MaskedSelfAttention(embed_size, 16, 8)
    out = attention(x)
    print(out.shape)
    fc = torch.nn.Linear(8, 65)
    out = fc(out)
    print(out.shape)

if __name__ == '__main__':
    test()