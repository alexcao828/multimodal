import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import matplotlib.pyplot as plt
import copy

chop_dim = 50
num_chops = 40

def insert_cls_tokens(input_ids, attention_masks, num_chops, chop_dim):
    for i in range(num_chops):
        if i == 0:
            new_input_ids = input_ids[:, 0:chop_dim]
            new_attention_masks = attention_masks[:, 0:chop_dim]
        else:
            start = chop_dim + (i-1)*(chop_dim-1)
            end = start + (chop_dim-1)

            temp1 = input_ids[:, 0:1]
            temp2 = input_ids[:, start:end]
            new_input_ids = torch.cat((new_input_ids, temp1, temp2), 1)

            temp1 = attention_masks[:, 0:1]
            temp2 = attention_masks[:, start:end]
            new_attention_masks = torch.cat((new_attention_masks, temp1, temp2), 1)
    return new_input_ids, new_attention_masks

def drop_empty_for_training(input_ids, attention_masks, labels):
    keep = []
    N = input_ids.size(dim=0)
    for i in range(N):
        if attention_masks[i, 1] == 1:
            keep.append(i)
    keep = torch.Tensor(keep)
    keep = keep.long()
    return input_ids[keep], attention_masks[keep], labels[keep]

train = pd.read_csv('train.csv')
num_train = train.shape[0]
train_labels = pd.get_dummies(train[['section']]).values
train_labels = np.argmax(train_labels, 1)
train_labels = np.repeat(train_labels, num_chops)
train_labels = torch.tensor(train_labels)
train_articles = train['article'].tolist()
train_articles = [s if isinstance(s, str) else '' for s in train_articles]

val = pd.read_csv('val.csv')
num_val = val.shape[0]
val_labels = pd.get_dummies(val[['section']]).values
val_labels = np.argmax(val_labels, 1)
val_labels = np.repeat(val_labels, num_chops)
val_labels = torch.tensor(val_labels)
val_articles= val['article'].tolist()
val_articles = [s if isinstance(s, str) else '' for s in val_articles]

test = pd.read_csv('test.csv')
num_test = test.shape[0]
test_labels = pd.get_dummies(test[['section']]).values
test_labels = np.argmax(test_labels, 1)
test_labels = np.repeat(test_labels, num_chops)
test_labels = torch.tensor(test_labels)
test_articles= test['article'].tolist()
test_articles = [s if isinstance(s, str) else '' for s in test_articles]

# use cuda to run this program on GPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# load bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_encodings = tokenizer(train_articles, truncation=True, padding='max_length', return_tensors='pt', max_length=chop_dim*num_chops)
train_input_ids = train_encodings['input_ids']
train_attention_masks = train_encodings['attention_mask']
train_input_ids, train_attention_masks = insert_cls_tokens(train_input_ids, train_attention_masks, num_chops=num_chops, chop_dim=chop_dim)

train_input_ids = torch.reshape(train_input_ids, (-1, chop_dim))
train_attention_masks = torch.reshape(train_attention_masks, (-1, chop_dim))
train_input_ids_wo_empty, train_attention_masks_wo_empty, train_labels_wo_empty = drop_empty_for_training(train_input_ids, train_attention_masks, train_labels)

val_encodings = tokenizer(val_articles, truncation=True, padding='max_length', return_tensors='pt', max_length=chop_dim*num_chops)
val_input_ids = val_encodings['input_ids']
val_attention_masks = val_encodings['attention_mask']
val_input_ids, val_attention_masks = insert_cls_tokens(val_input_ids, val_attention_masks, num_chops=num_chops, chop_dim=chop_dim)

val_input_ids = torch.reshape(val_input_ids, (-1, chop_dim))
val_attention_masks = torch.reshape(val_attention_masks, (-1, chop_dim))
val_input_ids_wo_empty, val_attention_masks_wo_empty, val_labels_wo_empty = drop_empty_for_training(val_input_ids, val_attention_masks, val_labels)

test_encodings = tokenizer(test_articles, truncation=True, padding='max_length', return_tensors='pt', max_length=chop_dim*num_chops)
test_input_ids = test_encodings['input_ids']
test_attention_masks = test_encodings['attention_mask']
test_input_ids, test_attention_masks = insert_cls_tokens(test_input_ids, test_attention_masks, num_chops=num_chops, chop_dim=chop_dim)

test_input_ids = torch.reshape(test_input_ids, (-1, chop_dim))
test_attention_masks = torch.reshape(test_attention_masks, (-1, chop_dim))

# splitting traing and validation dataset
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataset_wo_empty = TensorDataset(train_input_ids_wo_empty, train_attention_masks_wo_empty, train_labels_wo_empty)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_dataset_wo_empty = TensorDataset(val_input_ids_wo_empty, val_attention_masks_wo_empty, val_labels_wo_empty)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# load datasets into dataloaders
train_wo_empty_dataloader = DataLoader(train_dataset_wo_empty, shuffle=True, batch_size=128)
train_dataloader = DataLoader(train_dataset, batch_size=128)
val_wo_empty_dataloader = DataLoader(val_dataset_wo_empty, batch_size=128)
val_dataloader = DataLoader(val_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)

# setting the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=24,  # The number of output labels--2 for binary classification.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=True,  # Whether the model returns all hidden-states.
)
model = model.to(device)

# setting the epochs
num_epochs = 50
# setting for gradient descent
optimizer = AdamW(model.parameters(), lr=1e-6)
best_val_acc = 0
train_accs = []
val_accs = []
for epoch in range(num_epochs):
    y_pred = []
    y_true = []
    model.train()
    for batch in train_wo_empty_dataloader:
        # send batches to device (cpu or gpu)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        outputs = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(predictions.tolist())
        y_true.extend(b_labels.tolist())
    train_accs.append(accuracy_score(y_pred, y_true))

    y_pred = []
    y_true = []
    model.eval()
    for batch in val_wo_empty_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(predictions.tolist())
        y_true.extend(b_labels.tolist())
    val_accs.append(accuracy_score(y_pred, y_true))

    plt.figure(figsize=(12, 8))
    plt.plot(train_accs, 'bo-', label='train')
    plt.plot(val_accs, 'ro-', label='val')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc="lower right")
    plt.savefig('acc.png')
    plt.close()

    if val_accs[-1] > best_val_acc:
        best_val_acc = accuracy_score(y_pred, y_true)
        print('epoch =', epoch)
        print('best val acc = ', best_val_acc)
        print()
        best_model_wts = copy.deepcopy(model.state_dict())
model.load_state_dict(best_model_wts)

model.eval()
flag = 0
for batch in train_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)
    cls_hidden_state = outputs.hidden_states
    cls_hidden_state = cls_hidden_state[-1]
    cls_hidden_state = cls_hidden_state[:, 0, :]
    cls_hidden_state = cls_hidden_state.tolist()
    cls_hidden_state = np.asarray(cls_hidden_state)

    logits = outputs.logits
    logits = logits.tolist()
    logits = np.asarray(logits)
    if flag == 0:
        train_cls_hidden_states = cls_hidden_state
        train_logits = logits
        flag = 1
    else:
        train_cls_hidden_states = np.concatenate((train_cls_hidden_states, cls_hidden_state), axis=0)
        train_logits = np.concatenate((train_logits, logits), axis=0)
train_cls_hidden_states = np.reshape(train_cls_hidden_states, (num_train, num_chops, -1))
train_logits = np.reshape(train_logits, (num_train, num_chops, 24))
np.save('train_bodies_peripheral.npy', train_cls_hidden_states)
np.save('train_bodies_logits.npy', train_logits)

model.eval()
flag = 0
for batch in val_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)
    cls_hidden_state = outputs.hidden_states
    cls_hidden_state = cls_hidden_state[-1]
    cls_hidden_state = cls_hidden_state[:, 0, :]
    cls_hidden_state = cls_hidden_state.tolist()
    cls_hidden_state = np.asarray(cls_hidden_state)

    logits = outputs.logits
    logits = logits.tolist()
    logits = np.asarray(logits)
    if flag == 0:
        val_cls_hidden_states = cls_hidden_state
        val_logits = logits
        flag = 1
    else:
        val_cls_hidden_states = np.concatenate((val_cls_hidden_states, cls_hidden_state), axis=0)
        val_logits = np.concatenate((val_logits, logits), axis=0)
val_cls_hidden_states = np.reshape(val_cls_hidden_states, (num_val, num_chops, -1))
val_logits = np.reshape(val_logits, (num_val, num_chops, 24))
np.save('val_bodies_peripheral.npy', val_cls_hidden_states)
np.save('val_bodies_logits.npy', val_logits)

model.eval()
flag = 0
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)
    cls_hidden_state = outputs.hidden_states
    cls_hidden_state = cls_hidden_state[-1]
    cls_hidden_state = cls_hidden_state[:, 0, :]
    cls_hidden_state = cls_hidden_state.tolist()
    cls_hidden_state = np.asarray(cls_hidden_state)

    logits = outputs.logits
    logits = logits.tolist()
    logits = np.asarray(logits)
    if flag == 0:
        test_cls_hidden_states = cls_hidden_state
        test_logits = logits
        flag = 1
    else:
        test_cls_hidden_states = np.concatenate((test_cls_hidden_states, cls_hidden_state), axis=0)
        test_logits = np.concatenate((test_logits, logits), axis=0)
test_cls_hidden_states = np.reshape(test_cls_hidden_states, (num_test, num_chops, -1))
test_logits = np.reshape(test_logits, (num_test, num_chops, 24))
np.save('test_bodies_peripheral.npy', test_cls_hidden_states)
np.save('test_bodies_logits.npy', test_logits)

