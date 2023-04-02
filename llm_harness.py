import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import mlp
import wavenet
import dataset
from generator import generate_text
from torch.utils.tensorboard import SummaryWriter

def main():
    CONTEXT_LEN = 32
    learner_name = 'MLP'
    optimizer_name = 'Adam'
    train_max_iter = 40000
    batch_size = 64

    print('build_dataset() begin')
    xs, ys, char2index, index2char = dataset.build_dataset(dataset.get_shakespeare_text(), CONTEXT_LEN)
    print('build_dataset() end')
    print(f'X shape = {xs.shape}')
    print(f'Y shape = {ys.shape}')

    torch.manual_seed(42)

    alphabet_size = len(char2index)

    learners = {
        'MLP': (mlp.make_model, 0.01),
        'WaveNet': (wavenet.make_model, 0.01),
    }
    model_creator, lr = learners[learner_name]
    model = model_creator(alphabet_size, CONTEXT_LEN, embed_size=64, hidden_size=256)

    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
    }
    optimizer = optimizers[optimizer_name](model.parameters())

    writer = SummaryWriter(f'runs/{learner_name}_{optimizer_name}')
    writer.add_graph(model, xs[:1])

    losses = []
    curr_loss_estimate = 0
    loss_iter = 0
    iters_per_loss = 50

    loss_fn = torch.nn.CrossEntropyLoss()
    batch_count_per_epoch = (xs.shape[0] - 1) // batch_size + 1
    print(f'expected batch count to cover one epoch: {batch_count_per_epoch}')
    ud = {}

    for i in tqdm(range(train_max_iter)):
        optimizer.zero_grad()

        batch_idx = torch.randint(0, xs.shape[0], (batch_size,))
        loss = loss_fn(model(xs[batch_idx]), ys[batch_idx])

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
        param_count = 0
        for p in model.parameters():
            param_count += torch.numel(p)
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