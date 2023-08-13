"""load packages"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

MU = 1e-5

LR = 1e-5
num_epochs = 10000
batch_size = 128

T = 44
n_classes = 24
n_actions = 2
num_pixels = 7*7
image_arrival = 2
d_text_peripheral = 768
d_img_peripheral = 512

d_model = 500
d_fc = 100
n_heads = 8
d_k = 64
d_v = 64

"""load data"""
train_headlines = np.load('train_headlines_peripheral.npy')
train_abstracts = np.load('train_abstracts_peripheral.npy')
train_images = np.load('train_images_peripheral.npy')
train_images = np.transpose(train_images, (0, 2, 3, 1))
train_captions = np.load('train_captions_peripheral.npy')
train_bodies = np.load('train_bodies_peripheral.npy')
train_labels = pd.read_csv('train.csv')
train_labels = pd.get_dummies(train_labels[['section']]).values
train_labels = np.argmax(train_labels, 1)

val_headlines = np.load('val_headlines_peripheral.npy')
val_abstracts = np.load('val_abstracts_peripheral.npy')
val_images = np.load('val_images_peripheral.npy')
val_images = np.transpose(val_images, (0, 2, 3, 1))
val_captions = np.load('val_captions_peripheral.npy')
val_bodies = np.load('val_bodies_peripheral.npy')
val_labels = pd.read_csv('val.csv')
val_labels = pd.get_dummies(val_labels[['section']]).values
val_labels = np.argmax(val_labels, 1)

test_headlines = np.load('test_headlines_peripheral.npy')
test_abstracts = np.load('test_abstracts_peripheral.npy')
test_images = np.load('test_images_peripheral.npy')
test_images = np.transpose(test_images, (0, 2, 3, 1))
test_captions = np.load('test_captions_peripheral.npy')
test_bodies = np.load('test_bodies_peripheral.npy')
test_labels = pd.read_csv('test.csv')
test_labels = pd.get_dummies(test_labels[['section']]).values
test_labels = np.argmax(test_labels, 1)

"""device initialization"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU: ', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

''' Sinusoid position encoding table '''
def get_sinusoid_encoding_table(n_position, d_hid):
    def cal_angle(position, hid_idx ):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

"""model class"""
class omninet_early_stop(nn.Module):
    def __init__(self):
        super(omninet_early_stop, self).__init__()
        "parameters"
        "dimension of model- omninet paper"
        self.d_model = d_model
        self.d_fc = d_fc
        "number of heads in each transformer"
        self.n_heads = n_heads
        "dimenison of keys and values in transformers"
        self.d_k = d_k
        self.d_v = d_v

        self.n_classes = n_classes
        self.n_actions = n_actions
        self.num_pixels = num_pixels

        self.d_text_peripheral = d_text_peripheral
        self.d_img_peripheral = d_img_peripheral

        "activation functions"
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

        "peripherals"
        self.headline_peripheral = nn.Linear(self.d_text_peripheral, self.d_model)
        self.abstract_peripheral = nn.Linear(self.d_text_peripheral, self.d_model)
        self.image_peripheral = nn.Linear(self.d_img_peripheral, self.d_model)
        self.caption_peripheral = nn.Linear(self.d_text_peripheral, self.d_model)
        self.body_peripheral = nn.Linear(self.d_text_peripheral, self.d_model)

        "temporal cache transformer"
        "W^Q"
        self.w_qs_temporal = nn.Linear(self.d_model, self.n_heads * self.d_k )
        "W^K"
        self.w_ks_temporal = nn.Linear(self.d_model, self.n_heads * self.d_k )
        "W^V"
        self.w_vs_temporal = nn.Linear(self.d_model, self.n_heads * self.d_v)
        "initialize weights"
        nn.init.normal_(self.w_qs_temporal.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_k )))
        nn.init.normal_(self.w_ks_temporal.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_k )))
        nn.init.normal_(self.w_vs_temporal.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_v)))
        "layer norm and last fc layer"
        self.layer_norm_temporal = nn.LayerNorm(self.d_model)
        self.fc_temporal = nn.Linear(self.n_heads * self.d_v, self.d_model)
        nn.init.normal_(self.fc_temporal.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_v)))

        "gated transformer"
        "W^Q"
        self.w_qs_gated = nn.Linear(self.d_model, self.n_heads * self.d_k)
        "W^K"
        self.w_ks_gated = nn.Linear(self.d_model, self.n_heads * self.d_k)
        "W^V"
        self.w_vs_gated = nn.Linear(self.d_model, self.n_heads * self.d_v)
        "initialize weights"
        nn.init.normal_(self.w_qs_gated.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_k )))
        nn.init.normal_(self.w_ks_gated.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_k )))
        nn.init.normal_(self.w_vs_gated.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_v)))
        "layer norm and last fc layer"
        self.layer_norm_gated = nn.LayerNorm(self.d_model)
        self.fc_gated = nn.Linear(self.n_heads * self.d_v, self.d_model)
        nn.init.normal_(self.fc_gated.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.n_heads * self.d_v)))

        "fc layers for class arm"
        self.class_fc_1 = nn.Linear(self.d_model, self.d_fc)
        nn.init.normal_(self.class_fc_1.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_fc)))
        self.class_fc_2 = nn.Linear(self.d_fc, self.n_classes)
        nn.init.normal_(self.class_fc_2.weight, mean=0, std=np.sqrt(2.0 / (self.d_fc + self.n_classes)))

        "fc layers for act arm"
        self.act_fc_1 = nn.Linear(self.d_model, self.d_fc)
        nn.init.normal_(self.act_fc_1.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_fc)))
        self.act_fc_2 = nn.Linear(self.d_fc, self.n_actions)
        nn.init.normal_(self.act_fc_2.weight, mean=0, std=np.sqrt(2.0 / (self.d_fc + self.n_actions)))

        "temporal cache mask so no backward time leakage"
        self.MASK = torch.zeros((T, T), device=device)
        for i in range(T):
            for j in range(T):
                if j > i:
                    self.MASK[i, j] = -float("Inf")

        "gated mask so no backward time leakage for images"
        self.MASK2 = torch.ones((T, num_pixels), device=device)
        for i in range(T):
            if i < image_arrival:
                self.MASK2[i, :] = 0.0

    def forward(self, headlines, abstracts, images, captions, bodies):
        headlines= self.headline_peripheral(headlines)
        abstracts = self.abstract_peripheral(abstracts)
        images = self.image_peripheral(images)
        captions = self.caption_peripheral(captions)
        bodies = self.body_peripheral(bodies)

        spatial_cache = images.reshape(-1, num_pixels, self.d_model)

        headlines = torch.unsqueeze(headlines, dim=1)
        abstracts = torch.unsqueeze(abstracts, dim=1)
        images_mean = torch.mean(images, dim=(1,2))
        images_mean = torch.unsqueeze(images_mean, dim=1)
        captions = torch.unsqueeze(captions, dim=1)
        time_cache = torch.cat((headlines, abstracts, images_mean, captions, bodies), 1)
        pos_embed = get_sinusoid_encoding_table(T, self.d_model).to(device)
        time_cache = time_cache + pos_embed

        "init residuaal as time_cache"
        residual = time_cache

        "get batch_size and length of queries/keys/values of time_cache"
        sz_b, len_qkv, _ = time_cache.size()

        "get queries, keys, values matrices"
        q = self.w_qs_temporal(time_cache).view(sz_b, len_qkv, self.n_heads, self.d_k)
        k = self.w_ks_temporal(time_cache).view(sz_b, len_qkv, self.n_heads, self.d_k)
        v = self.w_vs_temporal(time_cache).view(sz_b, len_qkv, self.n_heads, self.d_v)

        "permute, reshape to (n*b) x lq x dk..."
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_qkv, self.d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_qkv, self.d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_qkv, self.d_v) # (n*b) x lv x dv

        "transformer/self-attention calculation"
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = self.softmax(attn / np.power(self.d_v, 0.5) + self.MASK)
        output = torch.bmm(attn, v)

        "permute, reshsape, feed output of transformer to fc layer for dim=d_model"
        output = output.view(self.n_heads, sz_b, len_qkv, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_qkv, -1) # b x lq x (n*dv)
        output = self.fc_temporal(output)
        "add residual and layer norm"
        output = self.layer_norm_temporal(output + residual)

        "permute attn"
        attn=attn.view(self.n_heads,sz_b,len_qkv,len_qkv).transpose(0,1)

        "gated transformer- see omninet alg 2"
        "get attention of image"
        temp_sel = attn[:, :, :, image_arrival:image_arrival+1]

        "expand temp_sel by 49 (image input is 7x7 subpixels)"
        temp_sel = temp_sel.unsqueeze(4).expand(sz_b, self.n_heads, len_qkv, 1, self.num_pixels).transpose(3, 4)
        k_gate = temp_sel.reshape(sz_b, self.n_heads, len_qkv, self.num_pixels*1)

        "init residual as previous output"
        residual = output

        "get queries (from output), keys (from spatial), values (from spatial) matrices"
        _, len_k, _ = spatial_cache.shape
        q = self.w_qs_gated(output).view(sz_b, len_qkv, self.n_heads, self.d_k)
        k = self.w_ks_gated(spatial_cache).view(sz_b, len_k, self.n_heads, self.d_k)
        v = self.w_vs_gated(spatial_cache).view(sz_b, len_k, self.n_heads, self.d_v)

        "permute, reshape to (n*b) x lq x dk..."
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_qkv, self.d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_v) # (n*b) x lv x dv

        "transformer/self-attention calculation for gated transformer- see eq 1 in paper"
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / np.power(self.d_v, 0.5)

        k_gate = k_gate.transpose(0, 1)
        k_gate=k_gate.reshape(self.n_heads*sz_b,len_qkv,self.num_pixels*1)
        attn=torch.mul(attn,k_gate)
        attn = self.softmax(attn)*self.MASK2
        output = torch.bmm(attn, v)

        "permute, reshsape, feed output of transformer to fc layer for dim=d_model"
        output = output.view(self.n_heads, sz_b, len_qkv, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_qkv, -1) # b x lq x (n*dv)
        output = self.fc_gated(output)
        "add residual and layer norm"
        output = self.layer_norm_gated(output + residual)

        "feed output through class fc layers to get class prediction logits of shape  b x lq x n_classes"
        class_logits = self.class_fc_1(output)
        class_logits = self.relu(class_logits)
        class_logits = self.class_fc_2(class_logits)

        "feed output through act fc layers to get act prediction logits of shape  b x lq x n_actions"
        act_logits = self.act_fc_1(output)
        act_logits = self.relu(act_logits)
        act_logits = self.act_fc_2(act_logits)
        return class_logits, act_logits

"create dataset (class to get next item) for torch"
class news_dataset(Dataset):
    def __init__(self,headlines, abstracts, images, captions, bodies, labels):
        self.headlines = headlines
        self.abstracts = abstracts
        self.images = images
        self.captions = captions
        self.bodies = bodies
        self.labels = labels
        
    def __getitem__(self,index):
        headline = self.headlines[index]
        headline = torch.tensor(headline)
        headline = headline.to(torch.float)

        abstract = self.abstracts[index]
        abstract = torch.tensor(abstract)
        abstract = abstract.to(torch.float)

        image = self.images[index]
        image = torch.tensor(image)
        image = image.to(torch.float)

        caption = self.captions[index]
        caption = torch.tensor(caption)
        caption = caption.to(torch.float)

        body = self.bodies[index]
        body = torch.tensor(body)
        body = body.to(torch.float)

        label = self.labels[index]
        return headline, abstract, image, caption, body, label

    def __len__(self):
        return len(self.labels)

"create dataloaders (get next batch) for torch"
training_dataset = news_dataset(train_headlines, train_abstracts, train_images, train_captions, train_bodies, train_labels)
val_dataset = news_dataset(val_headlines, val_abstracts, val_images, val_captions, val_bodies, val_labels)
test_dataset = news_dataset(test_headlines, test_abstracts, test_images, test_captions, test_bodies, test_labels)

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val':torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                    'test':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                   }

"create omninet model"
model = omninet_early_stop()

"function for determining ideal policy from predict class probs and labels"
def ideal_policy(class_probs, labels):
    "init 0s with shape (batch, T_end)"
    ideal = np.zeros((np.shape(class_probs)[0], np.shape(class_probs)[1]))
    "iterate through each sample"
    for i in range(np.shape(class_probs)[0]):
        "calc r for each sample for 1:T_end"
        r = np.log(1e-10 + class_probs[i, :, labels[i]]) - MU*np.arange(1,np.shape(class_probs)[1]+1)
        "time which r is maxed"
        here = np.argmax(r)
        "set ideal prob to classify to 1 from this point onward"
        ideal[i, here:] = 1
    return torch.tensor(ideal).to(torch.float)

"function for running actual policy output (action probs, predict class probs)"
def run_policy(act_probs, class_probs):
    "init guess per sample"
    g = np.zeros(np.shape(act_probs)[0])
    "init classifiction time c per sample"
    counters = np.zeros(np.shape(act_probs)[0])
    "iterate thru samples"
    for i in range(np.shape(act_probs)[0]):
        "probs to stop and classify"
        temp = act_probs[i, :, 1]
        "init class. time = 0"
        counter = 0
        "keep incrementing class. time until classification or T_end"
        while temp[counter] < 0.5:
            counter += 1
            if counter == T-1:
                break
        "calc guesss at class. time"
        g[i] = np.argmax(class_probs[i, counter, :])
        "track class. time in counter array"
        counters[i] = counter
    return g, counters

"main training function"
"input model, 2 losses for class and policy, optimizer, and number of epochs"
def train_model(model, criterion1, criterion2, optimizer, num_epochs=100):
    "track average reward and stopping time each phase"
    train_avg_acc = []
    val_avg_acc = []
    test_avg_acc = []

    train_avg_c = []
    val_avg_c = []
    test_avg_c = []

    train_all_cs = np.zeros((num_epochs, len(train_labels)))
    val_all_cs = np.zeros((num_epochs, len(val_labels)))
    test_all_cs = np.zeros((num_epochs, len(test_labels)))

    "iterate thru epochs"
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        "Each epoch has a training, val, and test phase"
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                "Set model to training mode"
                model.train()
            else:
                "Set model to evaluate mode"
                model.eval()

            ys = []
            gs = []
            all_cs = []

            if phase == 'val':
                valClassProbs = np.zeros((len(val_labels), T, 24))
                valActProbs = np.zeros((len(val_labels), T, 2))
                start = 0

            "Iterate over data"
            "each spatial, temporal, label batch"
            for h_batch, a_batch, i_batch, c_batch, b_batch, y_batch in dataloaders_dict[phase]:
                "send to device"
                h_batch = h_batch.to(device)
                a_batch = a_batch.to(device)
                i_batch = i_batch.to(device)
                c_batch = c_batch.to(device)
                b_batch = b_batch.to(device)
                y_batch = y_batch.to(device)
                
                "zero the parameter gradients"
                optimizer.zero_grad()

                "forward"
                "track history if only in train"
                with torch.set_grad_enabled(phase == 'train'):
                    "return logits of class and act probs from model"
                    class_p, act_p = model(h_batch, a_batch, i_batch, c_batch, b_batch)
                    "transform logits to probs"
                    class_p = F.softmax(class_p,dim=2)
                    act_p = F.softmax(act_p, dim=2)
                    
                    "backward + optimize only if in training phase"
                    if phase == 'train':
                        "get ideal policy ip and send to device"
                        ip = ideal_policy(class_p.cpu().detach().numpy(), y_batch.cpu().detach().numpy())
                        ip = ip.to(device)

                        "loss is 2 cross entropies, update"
                        loss = criterion1(class_p.contiguous().view(-1, n_classes), y_batch.repeat_interleave(T)) +criterion2(act_p.contiguous().view(-1, n_actions), ip.to(torch.long).contiguous().view(-1))
                        loss.backward()
                        optimizer.step()

                    "for both phases, run policy"
                    g, c = run_policy(act_p.cpu().detach().numpy(), class_p.cpu().detach().numpy())
                    "track acc and stopping time c"
                    ys.extend(y_batch.cpu().detach().numpy())
                    gs.extend(g)
                    all_cs.extend(c)

                    if phase == 'val':
                        stop = start + np.shape(act_p.cpu().detach().numpy())[0]
                        valClassProbs[start:stop, :, :] = class_p.cpu().detach().numpy()
                        valActProbs[start:stop, :, :] = act_p.cpu().detach().numpy()
                        start = stop

            "divide by num batches, print summary"
            avg_acc = accuracy_score(ys, gs)
            avg_c = np.mean(np.asarray(all_cs))
            values, counts = np.unique(all_cs, return_counts=True)
            print(phase+'--------------------')
            print('avg c = ', avg_c)
            print('avg acc = ', avg_acc)
            print('cs = ', values)
            print('cs hist = ', counts)
            print()

            if phase == 'train':
                train_avg_acc.append(avg_acc)
                train_avg_c.append(avg_c)
                train_all_cs[epoch, :] = np.asarray(all_cs)
            elif phase == 'val':
                val_avg_acc.append(avg_acc)
                val_avg_c.append(avg_c)
                val_all_cs[epoch, :] = np.asarray(all_cs)

                np.save('valClassProbs' + str(epoch) + '.npy', valClassProbs)
                np.save('valActProbs' + str(epoch) + '.npy', valActProbs)
            elif phase == 'test':
                test_avg_acc.append(avg_acc)
                test_avg_c.append(avg_c)
                test_all_cs[epoch, :] = np.asarray(all_cs)

                plt.figure(figsize=(12, 8))
                plt.plot(train_avg_acc, 'bs-', label='train')
                plt.plot(val_avg_acc, 'gs-', label='val')
                plt.plot(test_avg_acc, 'rs-', label='test')
                plt.legend(loc='lower right', prop={'size': 20})
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.savefig('acc.png')
                plt.close()

                plt.figure(figsize=(12, 8))
                plt.plot(train_avg_c, 'bs-', label='train')
                plt.plot(val_avg_c, 'gs-', label='val')
                plt.plot(test_avg_c, 'rs-', label='test')
                plt.legend(loc='upper right', prop={'size': 20})
                plt.xlabel('epoch')
                plt.ylabel('T')
                plt.savefig('T.png')
                plt.close()
        np.save('val_accs.npy', np.asarray(val_avg_acc))
        np.save('test_accs.npy', np.asarray(test_avg_acc))

        np.save('val_mean_Ts.npy', np.asarray(val_avg_c))
        np.save('test_mean_Ts.npy', np.asarray(test_avg_c))

        np.save('val_all_Ts.npy', val_all_cs)
        np.save('test_all_Ts.npy', test_all_cs)

"send model to device and print"
model.to(device)

"define optimizer, loss, and then train"
"Observe that all parameters are being optimized"
optim = optim.Adam(model.parameters(), lr=LR)
"class loss"
criterion1 = nn.CrossEntropyLoss()
"act loss"
criterion2 = nn.CrossEntropyLoss()
train_model(model, criterion1, criterion2, optim, num_epochs=num_epochs)
