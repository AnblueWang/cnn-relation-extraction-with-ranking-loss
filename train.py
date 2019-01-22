import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class RankingLossFunc(nn.Module):
    def __init__(self, args):
        super(RankingLossFunc, self).__init__()
        self.mPos = args.mPos
        self.mNeg = args.mNeg
        self.gamma = args.gamma
        self.cuda = args.cuda

    def forward(self, logit, target):
        val, ind = torch.topk(logit,2,dim=1) # top2 score
        noneOtherInd = target!=0 # not Other index
        rows = torch.tensor(list(range(len(logit)))) #row index
        part1 = logit[rows,target] # label score
#         if torch.min(part1) < -20:
#             part1 = self.gamma*(self.mPos-part1)
#         else:
        part1 = torch.log(1+torch.exp(self.gamma*(self.mPos-part1))) + torch.log(1+torch.exp(self.gamma*(-100+part1))) # positive loss
        predT = ind[:,0]==target
        predF = ind[:,0]!=target
#         if torch.max(val) > 20:
#             part2 = self.gamma*(self.mNeg+val)
#         else:
        part2 = torch.log(1+torch.exp(self.gamma*(self.mNeg+val)))+torch.log(1+torch.exp(self.gamma*(-100-val))) # negative loss
        part2 = torch.dot(predT.float(),part2[:,1])+torch.dot(predF.float(),part2[:,0])
        loss = torch.dot(noneOtherInd.float(),part1)+part2 # exclusive other loss
        return loss/len(target)


def getResult(logit,args):
    val, ind = torch.topk(logit,2,dim=1) #top2 score
    res = [ind[i][0] if ind[i][0]!=0 or ind[i][0]==0 and val[i][1] < 0 else ind[i][1] for i in range(len(logit))]
    res = torch.tensor(res)
    if args.cuda:
        res = res.cuda()
    return res

def computeF(result, target, args):
    precision,recall = 0,0
    
    correctPre = result.data == target.data
    nonOther = (target.data != 0).sum()
    predictNonOther = result.data != 0
    nonOtherCorrects = torch.dot(correctPre.float(),predictNonOther.float())
    nonOtherAcc = nonOtherCorrects/nonOther.float()
    
    classes = [(1,18),(2,11),(3,17),(4,14),(5,15),(6,7),(8,16),(9,12),(10,13)]
    for x in classes:
        predictPos = (result.data == x[0]) + (result.data == x[1])
        targetPos = (target.data == x[0]) + (target.data == x[1])
        truePos = torch.dot(predictPos.float(),targetPos.float())
        predictPosNum = predictPos.sum(dtype=torch.float32)
        if predictPosNum == 0:
            precision += 1;
        else:
            precision += truePos/predictPosNum
        targetPosNum = targetPos.sum(dtype=torch.float32)
        if targetPosNum == 0:
            recall += 1
        else:
            recall += truePos/targetPosNum
    
    precision /= 9
    recall /= 9
    return precision*100, recall*100, 200*precision*recall/(precision+recall)
        


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda(args.device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.max_norm)
    
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    print('training...')
    for epoch in range(1, args.epochs+1):
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr/epoch
        for batch in train_iter:
            feature, target, pos = batch.sentence, batch.relation, batch.pos_embed #(W,N) (N)
            feature.data.t_()
            pos.data.t_()
            
            if args.cuda:
                feature, target, pos = feature.cuda(), target.cuda(),pos.cuda()
                
            logit = model((feature,pos))
            criterion = RankingLossFunc(args)
            loss = criterion(logit, target)
#             loss = F.cross_entropy(logit, target)
#             print('loss:',loss)
            loss.backward()
            optimizer.step()
            
            steps += 1
            if steps % args.log_interval == 0:
                result = getResult(logit,args)
#                 result = torch.max(logit,1)[1].view(target.size())
                correctPre = result.data == target.data
                corrects = correctPre.sum()
                accuracy = corrects*100.0/batch.batch_size
                nonOther = (target.data != 0).sum()
                predictNonOther = result.data != 0
                nonOtherCorrects = torch.dot(correctPre.float(),predictNonOther.float())
                nonOtherAcc = 100*nonOtherCorrects/nonOther.float()
                sys.stdout.write('\rBatch[{}] - loss: {:.6f} acc: {:.4f}%({}/{}) nonOther acc: {:.4f}%（{}/{}）'.format(steps,
                                                                                  loss.data.item()/len(train_iter.dataset),
                                                                                        accuracy,
                                                                                        corrects,
                                                                                        batch.batch_size,
                                                                                        nonOtherAcc,
                                                                                        nonOtherCorrects,
                                                                                        nonOther))
            if steps % args.dev_interval == 0:
#                 print(torch.topk(logit,2,dim=1)[0])
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model,args.save_dir,'best',steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                        return
            elif steps % args.save_interval == 0:
                save(model,args.save_dir,'snapshot',steps)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0,0
    nonOther, nonOtherCorrects = 0,0
    for batch in data_iter:
        feature, target, pos = batch.sentence, batch.relation, batch.pos_embed #(W,N) (N) (2*W,N)
        feature.data.t_()
        pos.data.t_()

        if args.cuda:
            feature, target, pos = feature.cuda(), target.cuda(),pos.cuda()
        
        logit = model((feature,pos))
        criterion = RankingLossFunc(args)
        loss = criterion(logit, target)
#         loss = F.cross_entropy(logit, target)
        
        avg_loss += loss.data.item()
        result = getResult(logit,args)
#         result = torch.max(logit,1)[1].view(target.size())
        
        correctPre = result.view(target.size()).data == target.data
        corrects += correctPre.sum()
        nonOther += (target.data != 0).sum()
        predictNonOther = result.data != 0
        nonOtherCorrects += torch.dot(correctPre.float(),predictNonOther.float())
        
        pre, recall, F1 = computeF(result,target,args)
        
    
    size = len(data_iter.dataset)
    avg_loss /= size 
    accuracy = 100.0 * corrects/size
    nonOtherAcc = 100*nonOtherCorrects/nonOther.float()
    print('\nEvaluation - loss: {:.6f} macro precision:{:.4f}%, recall:{:.4f}%, F1: {:.4f}% acc: {:.4f}%({}/{}) nonOtherAcc: {:.4f}%({}/{})\n'.format(avg_loss,pre,recall,F1,accuracy,corrects,size,nonOtherAcc,nonOtherCorrects,nonOther))
    
    return nonOtherAcc

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir,save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix,steps)
    torch.save(model.state_dict(),save_path)