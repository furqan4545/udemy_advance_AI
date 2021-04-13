import sys
sys.path.insert(1, '/home/furqan/.pyenv/versions/3.8.5/lib/python3.8/site-packages')

import torch
from torch import nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):  # num_chars that we can predict
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size = (3, 3), padding = (1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2 = nn.Conv2d(128, 64, kernel_size = (3, 3), padding = (1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(1152, 64)  # 64 are the nums of features. 1152 is the result from last layer. 
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional= True, num_layers = 2, dropout=0.25)  # 64 inputs and 32 output filters
        self.output = nn.Linear(64, num_chars + 1) # why + 1
        # blc 0 index is for unknown value may be. 

    def forward(self, images, targets = None):  # targets can be none blc when u r in inference mode you don't need any target. 
        bs, c, h, w = images.size()
        # print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool_2(x)   # 1, 64, 18, 75  (batch_size, filters, height, width)
        # print(x.size())

        # Now from here we will add our RNN model. 
        # But before adding RNN model, we will perform some type of permutation. i.e. we will
        # bring width before height i.e. we will change the index position of width and height. 
        x = x.permute(0, 3, 1, 2)  # (1, 75, 64, 18)
        # print(x.size()) 
        x = x.view(bs, x.size(1), -1)
        # print(x.size())
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        # print(x.size()) # here we have 75 time stamps and for 75 timestamps we have 64 valeus. 
        # now we will add lstm gru model. 
        x, _ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size()) # (1, 75 , 20) -> so now we have 75 different time stamps and for each time stamps we have 20 outputs. 
        
        x = x.permute(1, 0, 2) # so your batch size will go to the middle. Your time stamp will be the first and ur values will be the last.  
        # print(x.size())
        # now we will check that if the user has specified the target or not. If he has specified then we are in training mode and we need to calcaulte loss.
        if targets is not None:
            # here we will calculate CTC loss 
            # ctc loss takes log_softmax. 
            log_softmax_values = F.log_softmax(x, 2) # 2 is the last value where you have all the classes. 
            # and now we need to specify two things. i.e. length of inputs and length of outputs
            input_lengths = torch.full(
                size = (bs, ), fill_value=log_softmax_values.size(0), dtype=torch.int32   
                # input length will have same size as batch_size. 
            )
            print(input_lengths)
            target_lengths = torch.full(
                size = (bs, ), fill_value=targets.size(1), dtype=torch.int32   
                # input length will have same size as batch_size. 
            )
            print(target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            ) 
            return x, loss

        return x, None


if __name__ == "__main__":
    cm = CaptchaModel(19) # 19 is then num of chars we are using in our model as a class label. 
    # here we are creating our own dataset 
    img = torch.rand(1, 3 ,75, 300)
    target = torch.randint(1, 20, (1, 5))
    x, loss = cm(img, target)