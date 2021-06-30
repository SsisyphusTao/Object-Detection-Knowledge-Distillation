import torch
from torch.autograd import Function, gradcheck
from torch.nn import Module
import torch.nn as nn
import dcn_op_v2
import math

class DeformableConv2DFunction(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_tensor, weight, bias, offset, mask, stride, pad, dilation, deformable_groups):
        ctx.stride_h = stride[0]
        ctx.stride_w = stride[1]
        ctx.pad_h = pad[0]
        ctx.pad_w = pad[1]
        ctx.dilation_h = dilation[0]
        ctx.dilation_w = dilation[1]
        ctx.deformable_groups = deformable_groups

        output = dcn_op_v2.forward(
            input_tensor,
            weight,
            bias,
            offset,
            mask,
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.deformable_groups
        )
        ctx.save_for_backward(input_tensor, weight, offset, mask, bias)
        return output
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *grad_outputs):
        input_tensor, weight, offset, mask, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias, grad_offset, grad_mask = dcn_op_v2.backward(
            input_tensor,
            weight,
            bias,
            offset,
            mask,
            grad_outputs[0],
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.deformable_groups
        )
        
        return grad_input, grad_weight, grad_bias, grad_offset, grad_mask, \
            None, None, None, None

class DeformableConv2DLayer(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 deformable_groups):
        super().__init__()
        def typexam(x):
            if type(x)==int:
                return (x, x)
            elif type(x)==tuple and len(x)==2:
                return x
            else:
                raise TypeError
        kernel_size = typexam(kernel_size)
        stride = typexam(stride)
        padding = typexam(padding)
        dilation = typexam(dilation)
        self.stride = stride
        self.pad = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          self.deformable_groups * 3 * kernel_size[0] * kernel_size[1],
                                          kernel_size=kernel_size,
                                          stride=self.stride,
                                          padding=self.pad,
                                          bias=True)
        self.reset_parameters(in_channels, kernel_size)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def reset_parameters(self, in_channels, kernel_size):
        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
    def forward(self, inputs):
        out = self.conv_offset_mask(inputs)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return DeformableConv2DFunction.apply(inputs, self.weight, self.bias, offset, mask, self.stride, self.pad, self.dilation, self.deformable_groups)

if __name__ == "__main__":
    deformable_groups = 1
    N, inC, inH, inW = 2, 2, 4, 4
    outC = 2
    kH, kW = 3, 3
    def check_gradient_dconv():

        t = torch.randn(N, inC, inH, inW).cuda()
        t.requires_grad = True

        offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda()
        # offset.data.zero_()
        # offset.data -= 0.5
        offset.requires_grad = True

        mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
        # mask.data.zero_()
        mask.requires_grad = True
        mask = torch.sigmoid(mask)

        weight = torch.randn(outC, inC, kH, kW).cuda()
        weight.requires_grad = True
        bias = torch.rand(outC).cuda()
        bias.requires_grad = True

        # DeformableConv2DFunction.apply(t, weight, bias, offset, mask, (1,1), (1,1), (1,1), deformable_groups)

        func = DeformableConv2DFunction.apply
        gradcheck(func, (t, weight, bias, offset, mask, (1,1), (1,1), (1,1), deformable_groups), eps=1e-3, atol=1e-3, rtol=1e-2)
    check_gradient_dconv()