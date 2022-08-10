import torch
import torch.onnx
import torch.nn as nn


# class Net(torch.nn.module):
#     def  __init__(self, n_feature, n_hidden):
#         Super(Net, self).__init__()
#         self.n_hidden = torch.nn.Conv()
#     def forward(self, x_layer):
#         x_layer = torch.relu(self.n_hidden(x_layer))
#         return x_layer
# model = Net(n_feature = 2,)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


model = nn.Sequential(
    nn.Conv2d(16, 16, 3),
    nn.ReLU(),
    Reshape(1,16,4,4) # add new idx 1 (2,1,16,4,4)
)
model.apply(weight_init)


batch_size = 1
export_onnx_file = "test3.onnx"			# 目的ONNX文件名
x = torch.randn(2, 16, 6, 6, requires_grad=True)
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})