import torch
from torch import nn
import torch.optim as optim
from nets import vgg_module, mobilenetv2_module
from penguin import getsingleimg


x, _ = getsingleimg()
vgg_test = vgg_module()
mobilenetv2_test = mobilenetv2_module()

vgg_test.eval()
vgg_test = vgg_test.cuda()
mobilenetv2_test.train()
mobilenetv2_test = mobilenetv2_test.cuda()
torch.backends.cudnn.benchmark = True

mbv2_s = mobilenetv2_test(x)[-1]
_, vgg_s = vgg_test(x)

l2_loss = nn.MSELoss()
output = l2_loss(mbv2_s, vgg_s)

first_block = []
for name, i in mobilenetv2_test.named_parameters():
    name = name.split('.')
    if 'features' in name[0] and int(name[1]) <= 7 :
        first_block.append(i)

optimizer = optim.SGD(first_block, lr=0.01, momentum=0.9,
                        weight_decay=5e-4)

output.backward()
optimizer.step()