import torch
import sys
import wavenet
import mlp
from dataset import get_shakespeare_text, tokenize

def generate_text(model, char2index, index2char, context_len, prompt):
    output = ''
    
    context = [char2index['<S>']] * context_len
    for c in prompt:
        context = context[1:] + [char2index[c]]
    
    for _ in range(300):
        pred = model(torch.tensor([context]))
        pred = torch.nn.functional.softmax(pred, dim=1)
        next_i = torch.multinomial(pred, 1, replacement=True).item()
        output += index2char[next_i]
        context = context[1:] + [next_i]
        
    return prompt + output

def main():
    learner_name = 'WaveNet'
    context_len = 32
    char2index, index2char = tokenize(get_shakespeare_text())
    model = wavenet.make_model(len(char2index), context_len, 64, 256)
    model.load_state_dict(torch.load(f'{learner_name}/weights.pt'))
    model.eval()

    print('prompt: ')
    prompt = ''
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    print(prompt)
    print('\n')
    output = generate_text(model, char2index, index2char, context_len, prompt)
    print('output: ')
    print(output)

if __name__ == '__main__':
    main()