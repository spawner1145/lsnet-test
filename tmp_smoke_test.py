import torch
import torch.nn as nn
from losses import ContrastiveLoss

print('Start smoke tests')

dev = torch.device('cpu')

# mem path
for batch in [2,8,32]:
    features = torch.randn(batch,128,requires_grad=True).to(dev)
    labels = torch.randint(0,5,(batch,)).to(dev)
    loss_fn = ContrastiveLoss(temperature=0.07,use_vq=False)
    loss,vq = loss_fn(features,labels)
    loss.backward()
    print('mem batch',batch,'ok, loss',loss.item(), 'grad norm', features.grad.norm().item())

# moco path
for batch in [2,4,16]:
    features = torch.randn(batch,64,requires_grad=True).to(dev)
    labels = torch.randint(0,5,(batch,)).to(dev)
    loss_fn = ContrastiveLoss(temperature=0.07,use_vq=False,use_queue=True,queue_size=32,dim=64)
    loss_fn.register_buffer('queue',torch.randn(64,32))
    loss_fn.queue = nn.functional.normalize(loss_fn.queue,dim=0)
    loss_fn.register_buffer('queue_ptr',torch.zeros(1,dtype=torch.long))
    loss_fn.queue_ptr[0]=0
    loss,vq = loss_fn(features,labels)
    loss.backward()
    print('moco batch',batch,'ok, loss',loss.item(),'grad norm',features.grad.norm().item())

# vq path
for batch in [2,8]:
    features = torch.randn(batch,32,requires_grad=True).to(dev)
    labels = torch.randint(0,5,(batch,)).to(dev)
    loss_fn = ContrastiveLoss(temperature=0.07,use_vq=True,vq_num_embeddings=64,vq_embedding_dim=32)
    loss,vq = loss_fn(features,labels)
    loss.backward()
    print('vq batch',batch,'ok, loss',loss.item(),'vq',vq.item(),'grad norm',features.grad.norm().item())

print('All smoke tests finished')
