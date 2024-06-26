import torch
from torch import nn
import torch.nn.functional as F


class ginar_cell(nn.Module):
    def __init__(self, num_id,in_size,emb_size,grap_size,dropout):
        super(ginar_cell, self).__init__()
        self.emb_size = emb_size
        self.num_id = num_id
        self.emb = nn.Linear(in_size,emb_size)
        self.emb2 = nn.Linear(num_id,num_id)

        self.att = InterpositionAttention(emb_size,emb_size, num_id,grap_size,dropout)
        self.linear1 = nn.Linear(num_id, num_id,bias=False)
        self.linear2 = nn.Linear(num_id, num_id, bias=True)

        self.layernorm = nn.LayerNorm([num_id])
        self.dropout = nn.Dropout(dropout)

        ### graph learning
        self.GL = nn.Parameter(torch.FloatTensor(num_id,grap_size))
        nn.init.kaiming_uniform_(self.GL)

        self.GL_linear = nn.Linear(grap_size, emb_size,bias=False)
        self.GL_linear2 = nn.Linear(emb_size * 2, emb_size * 2 ,bias=False)

    def forward(self, x, ct,graph_data):
        ### embeding
        x = F.leaky_relu(self.emb(x.transpose(-2, -1)))

        # Predefined graph
        graph_data1 = graph_data[0].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data1 = ginar_cell.calculate_laplacian_with_self_loop(graph_data1)
        graph_data2 = graph_data[1].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data2 = ginar_cell.calculate_laplacian_with_self_loop(graph_data2)

        ### Adaptive graph
        B,_,_ = x.shape
        GL_embed = self.GL.unsqueeze(0).expand(B, -1, -1)
        GL_embed = self.GL_linear2(torch.cat([self.emb2(self.att(x.transpose(-2, -1))).transpose(-2, -1), self.GL_linear(GL_embed)], dim=-1))
        graph_learn = torch.eye(self.num_id).to(x.device) + F.softmax(F.relu(GL_embed @ GL_embed.transpose(-2, -1)),dim=-1)

        ### GinAR cell
        x = x.transpose(-2, -1)
        x_new =     self.layernorm(self.dropout(self.linear1(self.att(x))) @ graph_learn + self.linear1(self.dropout(self.linear1(self.att(x))) @ graph_data1) @ graph_data2)
        ft = F.gelu(self.layernorm(self.dropout(self.linear2(self.att(x))) @ graph_learn + self.linear2(self.dropout(self.linear2(self.att(x))) @ graph_data1) @ graph_data2))
        rt = F.gelu(self.layernorm(self.dropout(self.linear2(self.att(x))) @ graph_learn + self.linear2(self.dropout(self.linear2(self.att(x))) @ graph_data1) @ graph_data2))
        ct = ft * ct + x_new - ft * x_new
        ht = rt * F.elu(ct) + x - rt * x
        return ht, ct

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian


class InterpositionAttention(nn.Module):
    def __init__(self, in_c, out_c,num_id,grap_size,dropout):
        """
        :param in_c: The input size of embedding
        :param out_c: The output size of embedding
        :param num_id: The number of variables
        :param grap_size: The node embedding size
        :param dropout: dropout
        """
        super(InterpositionAttention, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_id = num_id
        self.drop =dropout
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Parameter(torch.FloatTensor(size=(in_c, out_c)))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.FloatTensor(size=(2 * out_c, 1)))
        nn.init.xavier_uniform_(self.a)

        # leakyrelu
        self.leakyrelu = nn.LeakyReLU()
        self.GL = nn.Parameter(torch.FloatTensor(num_id, grap_size))
        nn.init.kaiming_uniform_(self.GL)

        # Graph generate
        self.GL2 = nn.Parameter(torch.FloatTensor(grap_size,num_id))
        nn.init.kaiming_uniform_(self.GL2)

    def forward(self, inp):
        """
        inp: input_fea [N, C]
        """
        ### graph
        adj = F.softmax(F.relu(self.GL @ self.GL.transpose(-2, -1)),dim=-1)
        # adj = F.softmax(F.relu(self.GL @ self.GL2),dim=-1)
        inp = inp.transpose(-2, -1)
        B, N = inp.size(0), inp.size(1)
        adj = adj + torch.eye(N, dtype=adj.dtype, device=adj.device)

        ### feature
        h = torch.matmul(inp, self.W) # [B,N,out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)
        # [B, N, N, 2 * out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [B,N, N, 1] => [B, N, N]

        ###Attention
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [B,N, N]
        attention = self.dropout(F.softmax(attention, dim=2))  #  [N, N]
        # attention = F.dropout(attention, self.drop, training=self.training)
        h_prime = torch.matmul(attention, self.dropout(h))  # [B,N, N].[N, out_features] => [B,N, out_features]
        h_prime = F.relu(h_prime)
        h_prime = h_prime.transpose(-2, -1)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
