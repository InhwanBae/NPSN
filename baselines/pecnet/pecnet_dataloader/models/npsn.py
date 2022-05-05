import torch
import torch.nn as nn


class GAT(nn.Module):
    def __init__(self, in_feat=2, out_feat=64, n_head=4, dropout=0.1, skip=True):
        super(GAT, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.n_head = n_head
        self.skip = skip
        self.w = nn.Parameter(torch.Tensor(n_head, in_feat, out_feat))
        self.a_src = nn.Parameter(torch.Tensor(n_head, out_feat, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, out_feat, 1))
        self.bias = nn.Parameter(torch.Tensor(out_feat))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        nn.init.constant_(self.bias, 0)

    def forward(self, h, mask):
        h_prime = h.unsqueeze(1) @ self.w
        attn_src = h_prime @ self.a_src
        attn_dst = h_prime @ self.a_dst
        attn = attn_src @ attn_dst.permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        attn = attn * mask if mask is not None else attn
        out = (attn @ h_prime).sum(dim=1) + self.bias
        if self.skip:
            out += h_prime.sum(dim=1)
        return out, attn


class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
        super(MLP, self).__init__()
        dims = (in_feat, ) + hid_feat + (out_feat, )

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x)
        return x


class NPSN(nn.Module):
    def __init__(self, t_obs=8, s=2, n=20):
        super(NPSN, self).__init__()
        self.s, self.n = s, n
        self.input_dim = t_obs * 2
        self.hidden_dim = self.input_dim * 4
        self.output_dim = s * n

        self.graph_attention = GAT(self.input_dim, self.hidden_dim)
        self.linear = MLP(self.hidden_dim, self.output_dim, activation=nn.ReLU())

    def forward(self, x, mask=None):
        node = x.reshape(x.size(0), x.size(1), -1)
        node, edge = self.graph_attention(node, mask)
        out = self.linear(node).reshape(x.size(0), x.size(1), self.n, -1)
        return out[..., 0:self.s].sigmoid().clamp(min=0.01, max=0.99)
