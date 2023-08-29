from torch.autograd import Function
import torch


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

    
class QuantAct(Function):
    @staticmethod
    def forward(ctx, x, n_bit):
        x_min = torch.min(x).detach()
        x_max = torch.max(x).detach()
        T = torch.max(torch.abs(x_min), torch.abs(x_max))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, -T, T)
        x_scale = x / T
        
        # Quant
        if n_bit == 32:
            pass
        elif n_bit == 1:
            x_scale = torch.sign(input)
        else:
            n = float(2 ** n_bit - 1)
            x_scale = torch.round(x_scale * n) / n

        # DeQuant
        x = x_scale * T
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class LSQAct(Function):
    @staticmethod
    def forward(ctx, x, scale, g, Qn, Qp):
        ctx.save_for_backward(x, scale)
        ctx.other = g, Qn, Qp
        x_scale = (x / scale).round().clamp(Qn, Qp)
        x = x_scale * scale
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other

        q_w = x / scale
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_scale = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_output * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_output

        return grad_weight, grad_scale, None, None, None
