import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, mlp_dropout:float=0., attn_dropout:float=0., qk_dropout:float=0.0, dropout_on='k', dropkey=False, mask_ratio:float=0.0):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, attn_dropout=attn_dropout, qk_dropout=qk_dropout, dropout_on=dropout_on, dropkey=dropkey, mask_ratio=mask_ratio)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


'''class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o'''

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, attn_dropout:float=0., qk_dropout:float=0.3, dropout_on='k', dropkey=False, mask_ratio:float=0.3):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.qk_dropout = qk_dropout

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(attn_dropout)

        self.mask_ratio = mask_ratio
        self.dropkey = dropkey

        # Dropout layers (you can apply dropout to 'q', 'k', or both)
        if dropout_on == 'q':
            self.dropout_q = nn.Dropout(qk_dropout)
            self.dropout_k = nn.Dropout(0.0)  # No dropout on key
        elif dropout_on == 'k':
            self.dropout_q = nn.Dropout(0.0)  # No dropout on query
            self.dropout_k = nn.Dropout(qk_dropout)
        elif dropout_on == 'qk':
            self.dropout_q = nn.Dropout(qk_dropout)
            self.dropout_k = nn.Dropout(qk_dropout)
        else: 
            self.dropout_q = nn.Dropout(0.0) 
            self.dropout_k = nn.Dropout(0.0) 
        
    def forward(self, x):
        b, n, f = x.size()
        # Apply dropout to input of query and key 
        xq = self.dropout_q(x)
        xk = self.dropout_k(x)
        # second way of doing that: 
        # m_q = torch.ones_like(x) * self.qk_dropout
        # m_k = torch.ones_like(x) * self.qk_dropout
        # xq = x * torch.bernoulli(m_q) * 0
        # xk = x * torch.bernoulli(m_k) * 0

        '''
        mask_q = torch.rand_like(x) > self.qk_dropout
        mask_k = torch.rand_like(x) > self.qk_dropout
        scale = 1.0 / (1.0 - self.qk_dropout)
        xq = x * mask_q * scale
        xk = x * mask_k * scale
        '''

        q = self.q(xq).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(xk).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d

        if self.dropkey:
            m_r = torch.ones_like(score) * self.mask_ratio
            score = score + torch.bernoulli(m_r) * -1e12

        norm_score = F.softmax(score, dim=-1)
        attn = torch.einsum("bhij, bhjf->bihf", norm_score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o
    

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b, n, f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)