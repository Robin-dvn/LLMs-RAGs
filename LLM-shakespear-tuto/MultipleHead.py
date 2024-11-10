from turtle import forward
from sympy import Q
import wget
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batchsize = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
nb_embed = 32
head_size = nb_embed
dropout= 0.
# ------------ #

#wget.download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
#chars du dataser
chars = sorted(list(set(text)))
vocab_size = len(chars)
#encoder decoder
stoi = {ch:i for (i,ch) in enumerate(chars)}
itos = {i:ch for (i,ch) in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

#train adn val split
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#dat loding
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batchsize,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.wk = nn.Linear(nb_embed,head_size, bias=False)
        self.wq = nn.Linear(nb_embed,head_size,bias=False)
        self.wv = nn.Linear(nb_embed,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self,x):
        B,T,C = x.shape
        q = self.wq(x) # B T headsize
        k = self.wk(x) # B T headsize

        wei = q @ k.transpose(-2,-1) * C**-0.5 # B T H * B H T = B T T 
        wei = wei.masked_fill(self.tril[:T,:T] == 0 , float('-inf')) # B T T 
        wei = F.softmax(wei,dim= -1 ) # B T T 
        wei = self.dropout(wei)

        v = self.wv(x)# B T headsize
        out = wei @ v # B T T * B T headsize = B T headsize
        return out 
class MultiHead(nn.Module):
    def __init__(self,nb_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(nb_heads)])
        self.proj = nn.Linear(nb_embed,nb_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self,nb_embed):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(nb_embed,nb_embed*4),
            nn.ReLU(),
            nn.Linear(nb_embed*4,nb_embed),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.seq(x)

class Block(nn.Module):
    def __init__(self,nb_embed,nb_head):
        super().__init__()

        self.sa_heads = MultiHead(nb_head,nb_embed//nb_head)
        self.ffw = FeedForward(nb_embed)
        self.ln1 = nn.LayerNorm(nb_embed)
        self.ln2 = nn.LayerNorm(nb_embed)
    
    def forward(self,x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size,nb_embed)
        self.position_embedding_table = nn.Embedding(block_size,nb_embed)
        self.blocks = nn.Sequential(
            Block(nb_embed,4),
            Block(nb_embed,4),
            Block(nb_embed,4),
            nn.LayerNorm(nb_embed)
        )
        self.lm_head = nn.Linear(nb_embed,vocab_size)


    def forward(self,idx,targets= None):
        #idx is B,T
        B,T = idx.shape
        token_embed = self.token_embedding_table(idx) # B T C
        pos_embed = self.position_embedding_table(torch.arange(T,device = device    )) # T C
        x = token_embed + pos_embed # broadcasring makes B T C

        x = x + self.blocks(x)
        logits = self.lm_head(x) # B T H * B H V =  B T V



        #modification de la shape pour la cross entropy
        if targets == None:
            loss = None
        else:
            B,T,V = logits.shape
            logits = logits.view(B*T,V)
            targets  = targets.view(B*T)

            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self, idx,max_new_tokens):
        #idx is size (B,T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1) #size (B,1)

            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    

model = MultiHeadAttention()
m = model.to(device)


#optimizer and training loop
optimizer = torch.optim.AdamW(m.parameters(),lr = learning_rate)

for iter in range(max_iters):
    xb,yb = get_batch('train')
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss = {losses['train']:.4f}  val loss = {losses['val']:.4f}")
    

    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate text
context = torch.zeros((1,1),dtype=torch.long,device = device)
print(decode(m.generate(idx=context,max_new_tokens=500)[0].tolist()))