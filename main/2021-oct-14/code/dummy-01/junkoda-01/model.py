from common import *
from einops.layers.torch import Rearrange

#----------------------------------------------------------------------------------

def mask_huber_loss(predict, truth, m, delta=0.1):
    loss = F.huber_loss(predict[m], truth[m], delta=delta)
    return loss

def mask_l1_loss(predict, truth, m):
    loss = F.l1_loss(predict[m], truth[m])
    return loss

def mask_smooth_l1_loss(predict, truth, m, beta=0.1):
    loss = F.smooth_l1_loss(predict[m], truth[m], beta=beta)
    return loss


class Net(nn.Module):
    def __init__(self,in_dim=10):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, 400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM( 2*400, 300, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM( 2*300, 200, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM( 2*200, 100, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2*100, 50),
            nn.SELU(),
        )
        self.pressure_in  = nn.Linear(50, 1)
        self.pressure_out = nn.Linear(50, 1)

        #----
        #initialisation : https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                #print(name,m)
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        batch_size = len(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.head(x)
        pressure_in  = self.pressure_in(x).reshape(batch_size,80)
        pressure_out = self.pressure_out(x).reshape(batch_size,80)
        return pressure_in, pressure_out


def run_check_net():
    batch_size = 10
    length = 80
    in_dim = 10
    x  = torch.randn((batch_size, length, in_dim-2))
    rc = torch.from_numpy(np.concatenate([
        np.random.choice([ 5, 20, 50],(batch_size, length, 1)),
        np.random.choice([10, 20, 50],(batch_size, length, 1)),
    ],2)).float()
    x  = torch.cat([x,rc], 2)

    net = Net(in_dim)
    pressure_in, pressure_out = net(x)
    print('x  :', x.shape)
    print('pressure_in  :', pressure_in.shape)
    print('pressure_out :', pressure_out.shape)

##################################################################################
if __name__ == '__main__':
    run_check_net()