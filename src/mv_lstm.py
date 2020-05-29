import torch
import torch.nn as nn

class MV_LSTM(nn.Module):
    def __init__(self, input_size, seq_length, num_output, n_hidden=200, n_layers=1, device="cpu"):
        super(MV_LSTM, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.device = device

        self.l_lstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=self.n_hidden,
                              num_layers=self.n_layers,
                              batch_first=True,
                              dropout=0.25)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, num_output)
        self.dropout = nn.Dropout(p=0.2)


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.randn(self.n_layers, batch_size, self.n_hidden).to(self.device)
        cell_state = torch.randn(self.n_layers, batch_size, self.n_hidden).to(self.device)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        linear_input = lstm_out.contiguous().view(batch_size,-1)
        output = self.l_linear(linear_input)
        output = self.dropout(output)
        return output
