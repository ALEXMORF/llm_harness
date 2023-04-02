import torch

class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

def make_model(alphabet_size, context_len, embed_size, hidden_size):
    model = torch.nn.Sequential(
        # batch_size, context_len,
        torch.nn.Embedding(alphabet_size, embed_size),
        # batch_size, context_len, embed_size
        View((context_len//2, embed_size*2)),
        # batch_size, context_len//2, embed_size*2
        torch.nn.Linear(embed_size*2, hidden_size), torch.nn.BatchNorm1d(context_len//2), torch.nn.ReLU(),
        # batch_size, context_len//2, hidden_size
        View((context_len//4, hidden_size*2)),
        # batch_size, context_len//4, hidden_size*2
        torch.nn.Linear(hidden_size*2, hidden_size), torch.nn.BatchNorm1d(context_len//4), torch.nn.ReLU(),
        # batch_size, context_len//4, hidden_size
        View((context_len//8, hidden_size*2)),
        # batch_size, context_len//8, hidden_size*2
        torch.nn.Linear(hidden_size*2, hidden_size), torch.nn.BatchNorm1d(context_len//8), torch.nn.ReLU(),
        # batch_size, context_len//8, hidden_size
        View((context_len//16, hidden_size*2)),
        # batch_size, context_len//16, hidden_size*2
        torch.nn.Linear(hidden_size*2, hidden_size), torch.nn.BatchNorm1d(context_len//16), torch.nn.ReLU(),
        # batch_size, context_len//16, hidden_size
        torch.nn.Flatten(),
        # batch_size, hidden_size*2
        torch.nn.Linear(hidden_size*2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(),
        # batch_size, hidden_size
        torch.nn.Linear(hidden_size, alphabet_size)
    )
    return model

def test():
    model = make_model(65, 32, 71, 91)
    preds = model(torch.tensor(64*[32*[1]]))
    print(preds.shape)

if __name__ == '__main__':
    print(f'running wavenet test code...')
    test()