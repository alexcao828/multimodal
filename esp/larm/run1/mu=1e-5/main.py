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

LR = 1e-6
num_epochs = 10000
batch_size = 128

T = 43
n_classes = 2
n_actions = 2
num_pixels = 7*7
image_arrival = 0
d_text_peripheral = 300
d_img_peripheral = 512

d_model = 300
d_fc = 100
n_heads = 8
d_k = 64
d_v = 64

"""load data"""
train_images = np.load('train_images_peripheral.npy')
train_images = np.transpose(train_images, (0, 2, 3, 1))
train_captions = np.load('trainGlove.npy')
train_labels = pd.read_csv('train.csv')
train_labels['wordsRoots'] = train_labels['words'].apply(lambda x: x[0:-9])
train_labels['labels'] = np.where(train_labels['roots'] == train_labels['wordsRoots'], 'match', 'notMatch')
train_labels = pd.get_dummies(train_labels[['labels']]).values

val_images = np.load('val_images_peripheral.npy')
val_images = np.transpose(val_images, (0, 2, 3, 1))
val_captions = np.load('valGlove.npy')
val_labels = pd.read_csv('val.csv')
val_labels['wordsRoots'] = val_labels['words'].apply(lambda x: x[0:-9])
val_labels['labels'] = np.where(val_labels['roots'] == val_labels['wordsRoots'], 'match', 'notMatch')
val_labels = pd.get_dummies(val_labels[['labels']]).values

test_images = np.load('test_images_peripheral.npy')
test_images = np.transpose(test_images, (0, 2, 3, 1))
test_captions = np.load('testGlove.npy')
test_labels = pd.read_csv('test.csv')
test_labels['wordsRoots'] = test_labels['words'].apply(lambda x: x[0:-9])
test_labels['labels'] = np.where(test_labels['roots'] == test_labels['wordsRoots'], 'match', 'notMatch')
test_labels = pd.get_dummies(test_labels[['labels']]).values

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

def fake_cumprod_exclusive(vb):
    """
    args:
        vb:  [hei x wid] Variable
          -> NOTE: we are lazy here so now it only supports cumprod along wid
    """
    # real_cumprod = torch.cumprod(vb.data, 1)
    vb = vb.unsqueeze(0)
    mul_mask_vb = torch.zeros([vb.size(2), vb.size(1), vb.size(2)], device=device).type_as(vb)
    for i in range(1, vb.size(2)):
       mul_mask_vb[i, :, 0:i] = 1
    add_mask_vb = 1 - mul_mask_vb
    vb = vb.expand_as(mul_mask_vb) * mul_mask_vb + add_mask_vb
    vb = torch.prod(vb, 2).transpose(0, 1)
    # print(real_cumprod - vb.data) # NOTE: checked, ==0
    return vb

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
        self.image_peripheral = nn.Linear(self.d_img_peripheral, self.d_model)
        self.caption_peripheral = nn.Linear(self.d_text_peripheral, self.d_model)

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

        "create dropout"
        self.dropout = nn.Dropout(0.9)

    def forward(self, images, captions):
        images = self.image_peripheral(images)
        captions = self.caption_peripheral(captions)

        spatial_cache = images.reshape(-1, num_pixels, self.d_model)

        images_mean = torch.mean(images, dim=(1,2))
        images_mean = torch.unsqueeze(images_mean, dim=1)
        time_cache = torch.cat((images_mean, captions), 1)
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

        A = self.softmax(act_logits)
        A = A[:, :, 1]
        A = self.dropout(A)*0.1
        B = fake_cumprod_exclusive(1 - A)
        A = B*A

        return class_logits, A, act_logits

"create dataset (class to get next item) for torch"
class espImageCaptions_dataset(Dataset):
    def __init__(self, images, captions, labels):
        self.images = images
        self.captions = captions
        self.labels = labels
        
    def __getitem__(self,index):
        image = self.images[index]
        image = torch.tensor(image)
        image = image.to(torch.float)

        caption = self.captions[index]
        caption = torch.tensor(caption)
        caption = caption.to(torch.float)

        label = self.labels[index]
        return image, caption, label

    def __len__(self):
        return len(self.labels[:, 0])

"create dataloaders (get next batch) for torch"
training_dataset = espImageCaptions_dataset(train_images, train_captions, train_labels)
val_dataset = espImageCaptions_dataset(val_images, val_captions, val_labels)
test_dataset = espImageCaptions_dataset(test_images, test_captions, test_labels)

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val':torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                    'test':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                   }

"create omninet model"
model = omninet_early_stop()

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
        while temp[counter] < np.random.uniform():
            counter += 1
            if counter == T-1:
                break
        "calc guesss at class. time"
        g[i] = np.argmax(class_probs[i, counter, :])
        "track class. time in counter array"
        counters[i] = counter
    return g, counters

"main training function"
"input model, optimizer, and number of epochs"
def train_model(model, optimizer, num_epochs=100):
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

            "Iterate over data"
            "each spatial, temporal, label batch"
            for i_batch, c_batch, y_batch in dataloaders_dict[phase]:
                "send to device"
                i_batch = i_batch.to(device)
                c_batch = c_batch.to(device)
                y_batch = y_batch.to(device)
                
                "zero the parameter gradients"
                optimizer.zero_grad()

                "forward"
                "track history if only in train"
                with torch.set_grad_enabled(phase == 'train'):
                    "return logits of class and act probs from model"
                    class_p, AA, act_p2 = model(i_batch, c_batch)
                    "transform logits to probs"
                    class_p = F.softmax(class_p,dim=2)
                    act_p2 = F.softmax(act_p2, dim=2)
                    
                    "backward + optimize only if in training phase"
                    if phase == 'train':
                        y_batch_tiled = y_batch.repeat(1, class_p.size(dim=1))
                        y_batch_tiled = y_batch_tiled.reshape(list(class_p.size()))
                        class_probs_loss = torch.sum(y_batch_tiled * class_p, 2)

                        loss = torch.mean(-torch.log(1e-10 + torch.sum(class_probs_loss *AA, 1)))
                        range1T = torch.range(1, class_p.size(dim=1), dtype=torch.float, device=device)
                        loss = loss + MU * torch.mean(torch.sum(range1T * AA, 1))

                        "loss update"
                        loss.backward()
                        optimizer.step()
                    "for both phases, run policy"
                    g, c = run_policy(act_p2.cpu().detach().numpy(), class_p.cpu().detach().numpy())
                    "track acc and stopping time c"
                    y2 = np.argmax(y_batch.cpu().detach().numpy(), 1)
                    ys.extend(y2)
                    gs.extend(g)
                    all_cs.extend(c)

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

        np.save('val_all_Ts.npy', val_all_cs[0:(epoch+1)])
        np.save('test_all_Ts.npy', test_all_cs[0:(epoch+1)])

"send model to device and print"
model.to(device)

"define optimizer, and then train"
"Observe that all parameters are being optimized"
optim = optim.Adam(model.parameters(), lr=LR)
train_model(model, optim, num_epochs=num_epochs)
