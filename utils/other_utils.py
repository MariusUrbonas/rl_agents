import torch

def to_pt(np_matrix, device, type='long'):
    if type == 'long':
        return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).to(device))
    elif type == 'float':
        return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).to(device))
    elif type =='bool':
        return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.BoolTensor).to(device))

