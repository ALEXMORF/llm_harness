import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import mlp
import wavenet
import Transformer
from dataset import get_shakespeare_text, tokenize
from generator import generate_text
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

def get_batch(dataset, context_len, batch_size, forTransformer = False):
    start_positions = torch.randint(0, dataset.shape[0]-context_len-1, (batch_size,))

    xs = None
    ys = None
    if forTransformer:
        xs = torch.zeros((batch_size, context_len), dtype=torch.long).to(device)
        ys = torch.zeros((batch_size, context_len), dtype=torch.long).to(device)
        for i in range(batch_size):
            pos = start_positions[i]
            xs[i] = dataset[pos:pos+context_len]
            ys[i] = dataset[pos+1:pos+context_len+1]
    else:
        xs = torch.zeros((batch_size, context_len), dtype=torch.long).to(device)
        ys = torch.zeros((batch_size,), dtype=torch.long).to(device)
        for batch_i in range(batch_size):
            start_pos = start_positions[batch_i]
            xs[batch_i] = dataset[start_pos:start_pos+context_len]
            ys[batch_i] = dataset[start_pos+context_len]
    return xs, ys

def get_model_param_count(model):
    param_count = 0
    for p in model.parameters():
        param_count += torch.numel(p)
    return param_count

def main():
    CONTEXT_LEN = 32
    learner_name = 'Transformer'
    optimizer_name = 'Adam'
    train_max_iter = 20000
    batch_size = 64

    torch.manual_seed(42)

    text = get_shakespeare_text()
    char2index, index2char = tokenize(text)
    dataset = torch.tensor([char2index[c] for c in text]).to(device)

    alphabet_size = len(char2index)

    learners = {
        'MLP': (mlp.make_model, 0.01),
        'WaveNet': (wavenet.make_model, 0.01),
        'Transformer': (Transformer.make_model, 0.01),
    }
    model_creator, lr = learners[learner_name]
    model = model_creator(alphabet_size, CONTEXT_LEN, embed_size=64, hidden_size=256).to(device)

    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
    }
    optimizer = optimizers[optimizer_name](model.parameters())

    print(f'model parameter count = {get_model_param_count(model)}')
    writer = SummaryWriter(f'runs/{learner_name}_{optimizer_name}')
    #writer.add_graph(model, xs[:1])

    losses = []
    curr_loss_estimate = 0
    loss_iter = 0
    iters_per_loss = 50

    loss_fn = torch.nn.CrossEntropyLoss()
    ud = {}

    for _ in tqdm(range(train_max_iter)):
        optimizer.zero_grad()

        xs, ys = get_batch(dataset, CONTEXT_LEN, batch_size, forTransformer=(learner_name == 'Transformer'))
        pred = model(xs)
        loss = None
        for context_pos_i in range(CONTEXT_LEN):
            pred_c = pred[:,context_pos_i,:]
            ys_c = ys[:,context_pos_i]
            if loss == None:
                loss = loss_fn(pred_c, ys_c)
            else:
                loss += loss_fn(pred_c, ys_c)
        loss /= CONTEXT_LEN

        curr_loss_estimate += (1.0 / iters_per_loss) * loss.item()
        loss_iter += 1
        if loss_iter == iters_per_loss:
            losses += [curr_loss_estimate]
            writer.add_scalar('Loss/train', curr_loss_estimate, len(losses)-1)
            writer.add_scalar('Likelihood/train', math.exp(-curr_loss_estimate), len(losses)-1)
            curr_loss_estimate = 0
            loss_iter = 0

        loss.backward()
        optimizer.step()
    model = model.eval()

    # report
    os.makedirs(f'{learner_name}', exist_ok=True)
    torch.save(model.state_dict(), f'{learner_name}/weights.pt')
    with open(f'{learner_name}/result.txt', 'w') as f:
        param_count = get_model_param_count(model)
        f.write(f'model parameter count = {param_count}\n')
        f.write(f'final loss = {losses[-1]}\n')

        prompt = ''
        output = generate_text(model, char2index, index2char, CONTEXT_LEN, prompt)
        f.write(f'prompt = "{prompt}"\n')
        f.write(f'output:\n')
        f.write(output)
    plt.plot(losses)
    plt.savefig(f'{learner_name}/losses.png')
    plt.clf()
    for layer_id, ratios in ud.items():
        log_ratios = torch.tensor(ratios).log10()
        plt.plot(log_ratios.detach(), label=f'layer {layer_id}')
    plt.legend()
    plt.savefig(f'{learner_name}/update-to-data.png')
    plt.clf()

if __name__ == '__main__':
    main()