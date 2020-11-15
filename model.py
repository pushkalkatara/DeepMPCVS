import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class VisualServoingLSTM(nn.Module):
  def __init__(self, rnn_type, vel_dims=6, lstm_units=6, layers=5, batch_size=1, seq_len=5):
    super(VisualServoingLSTM, self).__init__()
    self.vel_dims = vel_dims
    self.lstm_units = lstm_units
    self.layers = layers
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.f_interm= []
    self.v_interm= []
    if rnn_type == 'LSTM':
      self.lstm = nn.LSTM(vel_dims, lstm_units, layers, batch_first=True)
    elif rnn_type == 'GRU':
      self.lstm = nn.GRU(vel_dims, lstm_units, layers, batch_first=True)
    self.hidden = self.init_hidden(rnn_type)
    
  def init_hidden(self, rnn_type):
    cell = torch.randn(self.layers, self.batch_size, self.lstm_units)# 1,8,6
    cell = Variable(cell.cuda())
    if rnn_type == 'LSTM':
      hidden = torch.randn(self.layers, self.batch_size, self.lstm_units)# 1,8,6
      hidden = Variable(hidden.cuda())
      return hidden, cell
    else:
      return cell
  
  def reset_hidden(self):
    self.hidden = self.init_hidden('LSTM')

  def forward(self, vel, Lsx, Lsy):
    vels = None
    for i in range(self.seq_len):
        if i == 0:
            out, hidden = self.lstm(vel.view(1, 1, self.vel_dims), self.hidden)
            vels = out.unsqueeze(0) 
            # [1 x 1 x 6] X [5 x 1 x 6]
        else:
            out, hidden = self.lstm(out, hidden)
            vels = torch.cat([vels, out.unsqueeze(0)], dim=0)
        
        self.v_interm.append(out.data.cpu().numpy())

    L = torch.cat((Lsx, Lsy), -1)
    vels = vels.repeat(1, 1, 1, 2)
    f_hat = L*vels
    f_hat = torch.sum(f_hat, 0)
    f1, f2 = torch.split(f_hat, [6,6], -1)
    f1 = torch.sum(f1, -1).unsqueeze(-1)
    f2 = torch.sum(f2, -1).unsqueeze(-1)
    f_hat = torch.cat((f1, f2), -1)
    '''
    f_temp_5 = torch.sum(Lsx*out,-1).unsqueeze(-1)

    f_hat = torch.cat((torch.sum(Lsx*out,-1).unsqueeze(-1) , \
                  torch.sum(Lsy*out,-1).unsqueeze(-1)),-1)
    self.f_interm.append(f_hat.data.cpu().numpy())
    print(f_hat.shape)
    '''
    return f_hat