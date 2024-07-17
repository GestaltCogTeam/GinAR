import torch
from torch import nn, optim
import torch.nn.functional as F

from .ginar_cell import ginarCell


class GinAR(nn.Module):
    def __init__(self, input_len, num_id, out_len, in_size, emb_size,grap_size, layer_num,dropout,adj_mx):
        super(GinAR, self).__init__()

        ### basic parameter
        self.input_len = input_len
        self.out_len = out_len
        self.num_id = num_id
        self.layer_num = layer_num
        self.emb_size = emb_size
        self.graph_data = adj_mx

        ### encoder
        self.ginar_first = ginarCell(num_id,in_size,emb_size,grap_size,dropout)
        self.ginar_other = ginarCell(num_id, emb_size, emb_size, grap_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm([input_len,num_id])

        ### decoder
        self.decoder = nn.Conv2d(in_channels=layer_num,out_channels=out_len,kernel_size=(1,emb_size))
        self.output = nn.Conv2d(in_channels=out_len,out_channels=out_len,kernel_size=1)

    def forward(self, history_data):
        # Input [B,H,N,C]: B is batch size. N is the number of variables. H is the history length. C is the number of feature.
        # Output [B,L,N]: B is batch size. N is the number of variables. L is the future length

        x = history_data.transpose(-3, -1).transpose(-2, -1)
        B,C,L,N = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        graph_data = self.graph_data

        ### encoder
        final_result = 0.0
        for z in range(self.layer_num):
            result = 0.0
            ct = torch.zeros(B, self.emb_size, N).to(x.device)
            if z == 0:
                for j in range(self.input_len):
                    ht, ct = self.ginar_first(x[:,:,j,:], ct, graph_data)
                    if j == 0:
                        result = ht.unsqueeze(-2)
                    else:
                        result = torch.cat([result, ht.unsqueeze(-2)], dim=-2)
            else:
                for j in range(self.input_len):
                    ht, ct = self.ginar_other(x[:,:,j,:], ct, graph_data)
                    if j == 0:
                        result = ht.unsqueeze(-2)
                    else:
                        result = torch.cat([result, ht.unsqueeze(-2)], dim=-2)

            x = result.clone()
            result = result[:,:,-1,:]
            if z == 0:
                final_result = result.transpose(-2, -1).unsqueeze(1)
            else:
                final_result = torch.cat([final_result, result.transpose(-2, -1).unsqueeze(1)], dim=1)

        ### decoder
        x = self.dropout(self.decoder(final_result))
        x = self.output(x)
        return x.squeeze(-1)

