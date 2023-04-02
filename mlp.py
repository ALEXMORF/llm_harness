import torch

def make_model(alphabet_size, context_len, embed_size, hidden_size):
    model = torch.nn.Sequential(
        torch.nn.Embedding(alphabet_size, embed_size),
        torch.nn.Flatten(),
        torch.nn.Linear(embed_size*context_len, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, alphabet_size)
    )
    return model