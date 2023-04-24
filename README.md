# 2c-grad

**Goal:** Implement LeNet from scratch using c++ and cuda

![](./assets/mnist-lenet.png)

Using python we can implement the network architecture as follow:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
```

**TODO:**

1. A c++/cuda general tensor library that implements basic mathematical operations over multi-dimensional tensors,
	* [`tobias-mayer/autograd-cpp`](https://github.com/tobias-mayer/autograd-cpp)
	* [`sharan-dce/autograd`](https://github.com/sharan-dce/autograd)
1. An autograd engine that tracks the forward compute graph and can generate operations for the backward pass,
1. API to build the model and training 

