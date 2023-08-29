import torch
from torch.autograd import Function


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # IRNet
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output

        return grad_input, None, None
    

class STE(Function):
    @staticmethod
    def forward(ctx, x, bit_width):
        ctx.save_for_backward(x)
        if bit_width == 32:
            pass
        elif bit_width == 1:
            x = torch.sign(x)
        else:
            Qp = float(2 ** bit_width) -1
            Qn = - float(2 ** bit_width)
            x = torch.round(torch.clip(x, Qn, Qp))
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class LSQAct(Function):
    @staticmethod
    def forward(ctx, x, scale, g, Qn, Qp):
        ctx.save_for_backward(x, scale)
        ctx.other = g, Qn, Qp
        x_scale = (x / scale).round().clamp(Qn, Qp)
        x = x_scale * scale   # this should be checked
        return x_scale

    @staticmethod
    def bakward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other

        q_w = x / scale
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w>Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_scale = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())))
        grad_weight = indicate_middle * grad_output

        return grad_weight, grad_scale, None, None, None



class IR(Function):
    @staticmethod
    def forward(ctx, x, k, t):
        ctx.save_for_backward(x, k, t)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(x * t), 2)) * grad_output
        return grad_input, None, None
    
