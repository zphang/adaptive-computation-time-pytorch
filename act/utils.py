from torch.autograd import Variable


def maybe_cuda_var(x, cuda):
    """Helper for converting to a Variable"""
    x = Variable(x)
    if cuda:
        x = x.cuda()
    return x
