import torch

'''模型
    使用混合注意力代替多头注意力
    使用带有残差的卷积和LSTM
'''


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
