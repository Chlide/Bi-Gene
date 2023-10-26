
import generators
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
from time import perf_counter as t
from rbe_model import Encoder, Model
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import os
cpu_num = 48 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

device = 'cuda:0'

ref_num = 800000  # 严格对应 ref_num 数量 816566
ref_emb = np.load('ref_emb.800000.npy')
if ref_num > len(ref_emb):
    ref_num = len(ref_emb)
    print('ref_num', ref_num)

query_num = 200
query_emb = np.load('query_emb.800000.npy')[0:query_num]

dataset = TensorDataset(torch.tensor(ref_emb))
batch_size = 20480+4096*3 # 20480
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

dataset = TensorDataset(torch.tensor(query_emb))
batch_size = 20480+4096*3  # 20480
query_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('batch_size', batch_size)


@torch.no_grad()
def test(model):
    model.eval()

    from get_query import get_query
    labels_frag = [i for i in range(int(ref_num))]
    labels_subfrag = get_query('80w')[0:query_num]
    max_element = max(labels_subfrag)
    assert max_element < ref_num 

    # labels_subfrag = random.sample(range(100000), query_num)
    ref_emb = []
    for batch in tqdm(train_loader):
         batch = batch[0].to(device)
         b, _ = model(batch, batch)
         ref_emb.append(b)
    ref_emb = torch.cat(ref_emb, dim=0)
    
    query_emb = []
    for batch in tqdm(query_loader):
         batch = batch[0].to(device)
         b, _ = model(batch, batch)
         query_emb.append(b)
    query_emb = torch.cat(query_emb, dim=0)
    print(ref_emb, query_emb)

    test_query_emb = query_emb[0:len(query_emb)*0.5]
    test_labels_subfrag = labels_subfrag[0:len(labels_subfrag)*0.5]

    val_query_emb = query_emb[len(query_emb)*0.5:]
    val_labels_subfrag = labels_subfrag[len(labels_subfrag)*0.5:]


    def retrieval():
        from retrieval_metric import compute_distance, compute_retrieval
        sim_scores = compute_distance(val_query_emb.cpu().detach().numpy(), 
                                      ref_emb.cpu().detach().numpy(), mode='continuous',bs=150)
        compute_retrieval(a2b_sims=sim_scores, 
                            a_labels=val_labels_subfrag, 
                            b_labels=labels_frag,
                            mode='continuous',
                            topk=20)

        sim_scores = compute_distance(test_query_emb.cpu().detach().numpy(), 
                                      ref_emb.cpu().detach().numpy(), mode='continuous',bs=150)
        compute_retrieval(a2b_sims=sim_scores, 
                            a_labels=test_labels_subfrag, 
                            b_labels=labels_frag,
                            mode='continuous',
                            topk=20)
            
    retrieval()


queue=torch.Tensor([])
def train(model: Model, epoch):
    for batch in tqdm(train_loader):
        model.train()
        view1 = batch[0].to(device)
        optimizer.zero_grad()
        loss = 0

        if 1 == 1:
            b1, z = model(view1, view1)
            
            loss_bit_decorr = model.bt_loss(z, b1)
            loss = loss + loss_bit_decorr

            global queue
            loss_distill, queue = model.criterion(z, b1, temp=0.5, queue=queue, queue_len=10000)
            loss = loss + loss_distill


        if 1 == 1:
            batch_size = 512
            cos = nn.CosineSimilarity(dim=-1, eps=1e-8)
            distill_tau = 0.5

            rbe_1 = rbe_2 = b1
            full_precis = z
            losses = []
            for i in range(0, rbe_1.size(0), batch_size):
                z1T, z2T = rbe_1[i:i + batch_size], rbe_2[i:i + batch_size]
                student_top1_sim_pred = cos(z1T.unsqueeze(1), z2T.unsqueeze(0)) / distill_tau
                inputs = F.log_softmax(student_top1_sim_pred.fill_diagonal_(0.0), dim=-1)
                # print('student_top1_sim_pred', inputs)

                z1T = z2T = full_precis[i:i + batch_size]
                teacher_top1_sim_pred = cos(z1T.unsqueeze(1), z2T.unsqueeze(0)) / distill_tau
                targets = F.softmax(teacher_top1_sim_pred.fill_diagonal_(0.0), dim=-1)
                # print('teacher_top1_sim_pred',targets)

                ranking_loss = -(targets*inputs).nansum()  / targets.nansum()
                ranking_loss /= batch_size
                losses.append(ranking_loss)
            ranking_loss = torch.mean(torch.stack(losses))
            loss = loss + 0.5 * ranking_loss  # weight should not be too large in case of over-fitting

        print('total loss:', loss.item())
        loss.backward()
        optimizer.step()

    return loss


if __name__ == '__main__':   

    lr_rate = 0.0005
    model = Model(encoder=Encoder(hidden_channels=128), num_layers=1, tau=0.5).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_rate, weight_decay=0.00001)
    print('lr_rate', lr_rate)

    start = t()
    prev = start
    for epoch in tqdm(range(1, 400 + 1)):
        
        loss = train(model, epoch)
        print("=== Test ===")
        if epoch % 5 == 0:
            test(model)
            

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now


