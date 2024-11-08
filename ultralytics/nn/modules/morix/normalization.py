from functools import partial
import inspect, warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn

class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        # UserWarning: var(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel)
        if (x.size(-1) > 1):
            var_in = x.var(-1, keepdim=True)
        else:
            var_in = torch.zeros_like(mean_in)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias
    
    import torch

class SyncSNFunc(Function):
    @staticmethod
    def forward(ctx, in_data, scale_data, shift_data, mean_weight, var_weight,running_mean, running_var, eps, momentum, training):
        if in_data.is_cuda:
            ctx.eps =eps
            N, C, H, W = in_data.size()
            in_data = in_data.view(N, C, -1)
            mean_in = in_data.mean(-1, keepdim=True)
            var_in = in_data.var(-1, keepdim=True)

            mean_ln = mean_in.mean(1, keepdim=True)
            temp = var_in + mean_in ** 2
            var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

            if training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2

                sum_x = mean_bn ** 2 + var_bn
                dist.all_reduce(mean_bn)
                mean_bn /= dist.get_world_size()
                dist.all_reduce(sum_x)
                sum_x /= dist.get_world_size()
                var_bn = sum_x - mean_bn ** 2

                running_mean.mul_(momentum)
                running_mean.add_((1 - momentum) * mean_bn.data)
                running_var.mul_(momentum)
                running_var.add_((1 - momentum) * var_bn.data)

            else:
                mean_bn = torch.autograd.Variable(running_mean)
                var_bn = torch.autograd.Variable(running_var)

            softmax = nn.Softmax(0)
            mean_weight = softmax(mean_weight)
            var_weight = softmax(var_weight)

            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

            x_hat = (in_data - mean) / (var + ctx.eps).sqrt()
            x_hat = x_hat.view(N, C, H, W)
            out_data = x_hat * scale_data + shift_data

            ctx.save_for_backward(in_data.data, scale_data.data, x_hat.data, mean.data, var.data, mean_in.data, var_in.data,
                                  mean_ln.data, var_ln.data, mean_bn.data, var_bn.data, mean_weight.data, var_weight.data)
        else:
            raise RuntimeError('SyncSNFunc only support CUDA computation!')
        return out_data

    @staticmethod
    def backward(ctx, grad_outdata):
        if grad_outdata.is_cuda:

            in_data, scale_data, x_hat, mean, var, mean_in, var_in, mean_ln, var_ln, mean_bn, var_bn, \
                mean_weight, var_weight=  ctx.saved_tensors

            N, C, H, W = grad_outdata.size()
            scaleDiff = torch.sum(grad_outdata * x_hat,[0,2,3],keepdim=True)
            shiftDiff = torch.sum(grad_outdata,[0,2,3],keepdim=True)
            # dist.all_reduce(scaleDiff)
            # dist.all_reduce(shiftDiff)
            x_hatDiff = scale_data * grad_outdata

            meanDiff = -1 / (var.view(N,C) + ctx.eps).sqrt() * torch.sum(x_hatDiff,[2,3])

            varDiff = -0.5 / (var.view(N,C) + ctx.eps) * torch.sum(x_hatDiff * x_hat,[2,3])

            term1 = grad_outdata * scale_data / (var.view(N,C,1,1) + ctx.eps).sqrt()

            term21 = var_weight[0] * 2 * (in_data.view(N,C,H,W) - mean_in.view(N,C,1,1)) / (H*W) * varDiff.view(N,C,1,1)
            term22 = var_weight[1] * 2 * (in_data.view(N,C,H,W) - mean_ln.view(N,1,1,1)) / (C*H*W) * torch.sum(varDiff,[1]).view(N,1,1,1)
            term23_tmp = torch.sum(varDiff,[0]).view(1,C,1,1)
            dist.all_reduce(term23_tmp)
            term23 = var_weight[2] * 2 * (in_data.view(N,C,H,W) - mean_bn.view(1,C,1,1)) / (N*H*W) * term23_tmp / dist.get_world_size()

            term31 = mean_weight[0] * meanDiff.view(N,C,1,1) / H / W
            term32 = mean_weight[1] * torch.sum(meanDiff,[1]).view(N,1,1,1) / C  / H / W
            term33_tmp = torch.sum(meanDiff,[0]).view(1,C,1,1)
            dist.all_reduce(term33_tmp)
            term33 = mean_weight[2] * term33_tmp / N  / H / W / dist.get_world_size()

            inDiff =term1 + term21 + term22 + term23 + term31 + term32 + term33

            mw1_diff = torch.sum(meanDiff * mean_in.view(N,C))
            mw2_diff = torch.sum(meanDiff * mean_ln.view(N, 1))
            mw3_diff = torch.sum(meanDiff * mean_bn.view(1, C))

            dist.all_reduce(mw1_diff)
            # mw1_diff /= dist.get_world_size()
            dist.all_reduce(mw2_diff)
            # mw2_diff /= dist.get_world_size()
            dist.all_reduce(mw3_diff)
            # mw3_diff /= dist.get_world_size()

            vw1_diff = torch.sum(varDiff * var_in.view(N, C))
            vw2_diff = torch.sum(varDiff * var_ln.view(N, 1))
            vw3_diff = torch.sum(varDiff * var_bn.view(1, C))

            dist.all_reduce(vw1_diff)
            # vw1_diff /= dist.get_world_size()
            dist.all_reduce(vw2_diff)
            # vw2_diff /= dist.get_world_size()
            dist.all_reduce(vw3_diff)
            # vw3_diff /= dist.get_world_size()

            mean_weight_Diff = mean_weight
            var_weight_Diff = var_weight

            mean_weight_Diff[0] = mean_weight[0] * (mw1_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff- mean_weight[2] * mw3_diff )
            mean_weight_Diff[1] = mean_weight[1] * (mw2_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            mean_weight_Diff[2] = mean_weight[2] * (mw3_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            var_weight_Diff[0] = var_weight[0] * (vw1_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
            var_weight_Diff[1] = var_weight[1] * (vw2_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
            var_weight_Diff[2] = var_weight[2] * (vw3_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)


        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, scaleDiff, shiftDiff, mean_weight_Diff, var_weight_Diff, None, None, None, None, None

class SyncSwitchableNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9,last_gamma=False):
        super(SyncSwitchableNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma

        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.mean_weight = Parameter(torch.ones(3))
        self.var_weight = Parameter(torch.ones(3))

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, in_data):
        return SyncSNFunc.apply(
                    in_data, self.weight, self.bias, self.mean_weight, self.var_weight, self.running_mean, self.running_var, self.eps, self.momentum, self.training)
        
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if x.is_contiguous():
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

class FakeLayerNorm2d(nn.GroupNorm):
    """
    https://discuss.pytorch.org/t/groupnorm-num-groups-1-and-layernorm-are-not-equivalent/145468/2

    """
    def __init__(self, num_channels: int, eps: float = 0.00001, affine: bool = True, device=None, dtype=None):
        warnings.warn(
            "nn.GroupNorm(1, dim) are NOT equivent to nn.LayerNorm(dim)! \nRefers: \
                https://discuss.pytorch.org/t/groupnorm-num-groups-1-and-layernorm-are-not-equivalent/145468/2",
            DeprecationWarning,
        )
        super().__init__(1, num_channels, eps, affine, device, dtype)

SwitchNorm2dNoBatch = partial(SwitchNorm2d, using_bn=False)
SwitchNorm3dNoBatch = partial(SwitchNorm3d, using_bn=False)
