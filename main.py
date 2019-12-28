import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import re
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 20
    dim = 300
    out_channel = 100
    lr=0.001
    optimizer='Adam'
    dropout=0.5
    l2_constraint=3
    batch_size=50
    kernel_heights=[3,4,5]
    
    num_class=2

    len_sen = 60
config = config()

def make_data(file_path, label):
    raw_data = open(file_path, encoding='ISO-8859-1').readlines()
    pp_data = []
    pp_len = []
    for sen in raw_data:
        if label == "POS":
            temp = 0
        else:
            temp = 1
        pp_data.append((int(temp),re.sub('^\W+', "", sen.lower().strip())))
        pp_len.append(len(sen.split()))
        #pp_data.append( "<"+label +"> "+re.sub('^\W+', "", sen.lower().strip()))
    print("{}'s max_len is {}".format(label, max(pp_len)))
    print("{}'s min_len is {}".format(label, min(pp_len)))
    print("{}'s avg_len is {}".format(label, sum(pp_len)/len(pp_len)))
    return pp_data

def make_vocab(pd, nd):
    print("Pos data tokenizing...")
    #Assign an index number to a word
    word2ind = {}
    word2ind["<PAD>"]=0
    word2ind["<EOS>"]=1
    i = 2
    for _,sen in pd:
        for word in sen.split(): #except label
            if word not in word2ind.keys():
                word2ind[word] = i
                i+=1

    print("Neg data tokenizing...")
    for _,sen in nd:
        for word in sen.split():
            if word not in word2ind.keys():
                word2ind[word] = i
                i+=1

    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Word Vocabulary size")
    print(len(word2ind))
    print()
    return word2ind, ind2word

def make_vocab(pd, nd):
    print("Pos data tokenizing...")
    #Assign an index number to a word
    word2ind = {}
    word2ind["<PAD>"]=0
    word2ind["<EOS>"]=1
    i = 2
    for _,sen in pd:
        for word in sen.split(): #except label
            if word not in word2ind.keys():
                word2ind[word] = i
                i+=1

    print("Neg data tokenizing...")
    for _,sen in nd:
        for word in sen.split():
            if word not in word2ind.keys():
                word2ind[word] = i
                i+=1

    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Word Vocabulary size")
    print(len(word2ind))
    print()
    return word2ind, ind2word

def make_idx_sen(data, word2ind, len_sen = config.len_sen):
    new_data = []
    for label, sen in data:
        idx_sen = []
        for word in sen.split():
            if word in word2ind.keys():
                idx_sen.append(word2ind[word])
        idx_sen.append(word2ind['<EOS>'])
        while len(idx_sen)<len_sen:
            idx_sen.append(word2ind['<PAD>'])
        new_data.append((label, idx_sen))
    return new_data
            
            

class textCNN(nn.Module):
    def __init__(self, w2i,i2w,model_type, pre = None, dim=config.dim, num_class=config.num_class, kernel_heights=config.kernel_heights, len_sen = config.len_sen, out_channel = config.out_channel):
        super().__init__()
        self.word2ind = w2i
        self.ind2word = i2w
        self.model_type = model_type

        if model_type =='random':
            print("random type")
            self.embedding = nn.Embedding(len(w2i),dim)
        elif model_type == "static":
            print("static type")
            weights = []
            for ind in self.ind2word:
                word = self.ind2word[ind]
                if word in pre.vocab:
                    weights.append(pre.word_vec(word))
                else:
                    weights.append(np.random.uniform(-0.01, 0.01, dim).astype("float32"))
            weights = torch.FloatTensor(weights)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        elif model_type == "non-static":
            print("non-static type")
            weights = []
            for ind in self.ind2word:
                word = self.ind2word[ind]
                if word in pre.vocab:
                    weights.append(pre.word_vec(word))
                else:
                    weights.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
            weights = torch.FloatTensor(weights)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            print("Multi Channel type")
            weights = []
            for ind in self.ind2word:
                word = self.ind2word[ind]
                if word in pre.vocab:
                    weights.append(pre.word_vec(word))
                else:
                    weights.append(np.random.uniform(-0.01, 0.01, dim).astype("float32"))
            weights = torch.FloatTensor(weights)
            self.training_embedding = nn.Embedding.from_pretrained(weights, freeze=True)
            self.no_training_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        
        if model_type == 'multi_channel':
            self.c_3 = nn.Conv2d(2, out_channel, (kernel_heights[0],dim), stride=1, padding=0)
            self.c_4 = nn.Conv2d(2, out_channel, (kernel_heights[1],dim), stride=1, padding=0)
            self.c_5 = nn.Conv2d(2, out_channel, (kernel_heights[2],dim), stride=1, padding=0)

        else:
            self.c_3 = nn.Conv2d(1, out_channel, (kernel_heights[0],dim), stride=1, padding=0)
            self.c_4 = nn.Conv2d(1, out_channel, (kernel_heights[1],dim), stride=1, padding=0)
            self.c_5 = nn.Conv2d(1, out_channel, (kernel_heights[2],dim), stride=1, padding=0)

        self.maxpool_3 = nn.MaxPool1d((len_sen-kernel_heights[0]+1))
        self.maxpool_4 = nn.MaxPool1d((len_sen-kernel_heights[1]+1))
        self.maxpool_5 = nn.MaxPool1d((len_sen-kernel_heights[2]+1))

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(kernel_heights)*out_channel, num_class)


    def forward(self, sen):
        tensor_sen = torch.tensor(sen).to(config.device)
        if self.model_type is not "multi_channel":
            embedded = self.embedding(tensor_sen.to(config.device)).to(config.device) #bs, len, dim
            embedded = embedded.unsqueeze(1)
            #bs, len, dim --> bs, 1 len, dim
        else:
            train_embedded = self.training_embedding(tensor_sen.to(config.device)).to(config.device)        #bs, len, dim
            no_train_embedded = self.no_training_embedding(tensor_sen.to(config.device)).to(config.device)  #bs, len, dim
            embedded = torch.stack((train_embedded, no_train_embedded), dim=1)                              #bs, 2, len, dim
        
        result1 = self.dropout(self.maxpool_3(F.relu(self.c_3(embedded).squeeze(-1))).squeeze(-1))
        result2 = self.dropout(self.maxpool_4(F.relu(self.c_4(embedded).squeeze(-1))).squeeze(-1))
        result3 = self.dropout(self.maxpool_5(F.relu(self.c_5(embedded).squeeze(-1))).squeeze(-1))  #bs, 100
        concat = torch.cat((result1,result2,result3), dim=1)

        result = self.fc(concat)            #bs,num_class
        output = torch.softmax(result, dim=1)
        return output                       #bs,num_class
        

def train(model, data, batch_size=config.batch_size, epochs=config.epoch):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optim = torch.optim.Adam(model.parameters(),lr = config.lr)

    model.train()
    start = time.time()

    rands = np.arange(len(data))
    np.random.shuffle(rands)

    
    all_label = []
    all_sen = []
    for label, sen in data:
        all_label.append(label)
        all_sen.append(sen)
    tensor_label = torch.tensor(all_label).to(config.device)
    tensor_sen = torch.tensor(all_sen).to(config.device)

    tensor_train_label = tensor_label[rands[:9000]]
    tensor_train_sen = tensor_sen[rands[:9000]]

    tensor_test_label = tensor_label[rands[9000:]] 
    tensor_test_sen = tensor_sen[rands[9000:]] 

    batch_num = int(len(tensor_train_label)/ batch_size)

    for epoch in range(epochs):
        batch_label = []
        batch_sen = []
        epoch_loss = 0 
        for b in range(batch_num-1):
            if b == batch_num:
                batch_label.append(tensor_label[b*batch_size:])
                batch_sen.append(tensor_sen[b*batch_size:])
            else:
                batch_label.append(tensor_label[b*batch_size:(b+1)*batch_size])
                batch_sen.append(tensor_sen[b*batch_size:(b+1)*batch_size])
        for i in range(len(batch_label)):
            optim.zero_grad()
            label = batch_label[i].clone().detach().to(config.device)
            sen = batch_sen[i].clone().detach().to(config.device) #50, len
            preds = model(sen) #50,2

            loss= F.cross_entropy(preds.to(config.device), label) + torch.abs(config.l2_constraint-torch.norm(model.c_3.weight, p=2)) + torch.abs(config.l2_constraint-torch.norm(model.c_4.weight, p=2)) + torch.abs(config.l2_constraint-torch.norm(model.c_5.weight, p=2)) + torch.abs(config.l2_constraint-torch.norm(model.fc.weight, p=2))

            loss.backward()
            optim.step()

            epoch_loss += loss.item()
        loss_avg = epoch_loss / batch_size
        print("time = %dm, epoch %d, loss = %.3f, %ds per epoch" % ((time.time() - start) // 60, epoch + 1, loss_avg, (time.time() - start)/(epoch+1)))

        #test
        corr = 0
        tot = len(tensor_test_label)
        for i in range(len(tensor_test_label)):
            test_label = tensor_test_label[i].to(config.device)
            test_sen = tensor_test_sen[i].to(config.device)

            preds = model(test_sen.unsqueeze(0)) #50,2
            preds_label = torch.argmax(preds, dim = 1)
            if preds_label.item() == test_label.item():
                corr += 1
        print("corr/tot:{}/{}".format(corr,tot))
        print("corr rate:{}".format(corr/tot))

        
def main():
    
    pos_data= make_data('./rt-polaritydata(MR)/rt-polaritydata/rt-polarity.pos.txt', "POS") #0
    neg_data= make_data('./rt-polaritydata(MR)/rt-polaritydata/rt-polarity.neg.txt', "NEG") #1
    all_data = pos_data + neg_data
    
    word2ind, ind2word = make_vocab(pos_data, neg_data)
    
    all_idx_data = make_idx_sen(all_data, word2ind)
    
    pre = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    
    random_model = textCNN(word2ind,ind2word, model_type='random').to(config.device)
    static_model = textCNN(word2ind,ind2word, model_type='static', pre = pre).to(config.device)
    non_static_model = textCNN(word2ind,ind2word, model_type='non-static', pre = pre).to(config.device)
    mc_model = textCNN(word2ind,ind2word, model_type='multi_channel', pre = pre).to(config.device)
    
    train(random_model,all_idx_data) #10epoch
    train(static_model,all_idx_data) #10epoch
    train(non_static_model,all_idx_data) #10epoch
    train(mc_model,all_idx_data) #10epoch
    
main()