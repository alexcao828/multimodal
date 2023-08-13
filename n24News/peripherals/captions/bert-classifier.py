import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
train_labels = pd.get_dummies(train[['section']]).values
train_labels = np.argmax(train_labels, 1)
train_captions = train['caption'].tolist()
train_captions = [s if isinstance(s, str) else '' for s in train_captions]

val = pd.read_csv('val.csv')
val_labels = pd.get_dummies(val[['section']]).values
val_labels = np.argmax(val_labels, 1)
val_captions = val['caption'].tolist()
val_captions = [s if isinstance(s, str) else '' for s in val_captions]

test = pd.read_csv('test.csv')
test_labels = pd.get_dummies(test[['section']]).values
test_labels = np.argmax(test_labels, 1)
test_captions = test['caption'].tolist()
test_captions = [s if isinstance(s, str) else '' for s in test_captions]

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

train_encodings = tokenizer(train_captions, truncation=True, padding='max_length', return_tensors='pt', max_length=100)
train_input_ids = train_encodings['input_ids']
train_attention_masks = train_encodings['attention_mask']

val_encodings = tokenizer(val_captions, truncation=True, padding='max_length', return_tensors='pt', max_length=100)
val_input_ids = val_encodings['input_ids']
val_attention_masks = val_encodings['attention_mask']

test_encodings = tokenizer(test_captions, truncation=True, padding='max_length', return_tensors='pt', max_length=100)
test_input_ids = test_encodings['input_ids']
test_attention_masks = test_encodings['attention_mask']

# splitting traing and validation dataset
train_dataset = TensorDataset(train_input_ids, train_attention_masks, torch.tensor(train_labels))
val_dataset = TensorDataset(val_input_ids, val_attention_masks, torch.tensor(val_labels))
test_dataset = TensorDataset(test_input_ids, test_attention_masks, torch.tensor(test_labels))

# load datasets into dataloaders
train_shuffled_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=100)
train_ordered_dataloader = DataLoader(train_dataset, batch_size=100)
val_dataloader = DataLoader(val_dataset, batch_size=100)
test_dataloader = DataLoader(test_dataset, batch_size=100)

# setting the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=np.amax(train_labels)+1,  # The number of output labels--2 for binary classification.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=True,  # Whether the model returns all hidden-states.
)
model = model.to(device)

# setting the epochs
num_epochs = 200
# setting for gradient descent
optimizer = AdamW(model.parameters(), lr=1e-6)
best_val_acc = 0
train_accs = []
val_accs = []
for epoch in range(num_epochs):
    y_pred = []
    y_true = []
    model.train()
    for batch in train_shuffled_dataloader:
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
        print('best val acc = ', best_val_acc)
        print()

        model.eval()
        flag = 0
        for batch in train_ordered_dataloader:
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
            if flag == 0:
                train_cls_hidden_states = cls_hidden_state
                flag = 1
            else:
                train_cls_hidden_states = np.concatenate((train_cls_hidden_states, cls_hidden_state), axis=0)
        np.save('train_captions_peripheral.npy', train_cls_hidden_states)

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
            if flag == 0:
                val_cls_hidden_states = cls_hidden_state
                flag = 1
            else:
                val_cls_hidden_states = np.concatenate((val_cls_hidden_states, cls_hidden_state), axis=0)
        np.save('val_captions_peripheral.npy', val_cls_hidden_states)

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
            if flag == 0:
                test_cls_hidden_states = cls_hidden_state
                flag = 1
            else:
                test_cls_hidden_states = np.concatenate((test_cls_hidden_states, cls_hidden_state), axis=0)
        np.save('test_captions_peripheral.npy', test_cls_hidden_states)

