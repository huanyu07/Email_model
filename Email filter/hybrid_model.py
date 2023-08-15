import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'ham': 0, 'spam': 1}

batch_size = 16

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df_X, df_Y):
        self.labels = [labels[label] for label in df_Y['Label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length = 512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df_X['Email']]

    def classes(self):
        return self.labels
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
class Hybrid(nn.Module):
    def __init__(self, dropout=0.5):
        super(Hybrid, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.svm = nn.Linear(768, 2)
        self.relu = nn.ReLU()
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.svm(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def get_data():
    data = pd.read_csv("data/data_produced.csv", encoding='utf-8')
    X_train, X_test, Y_train, Y_test = train_test_split(data, data, test_size=0.3, random_state=0, shuffle=True)
    return X_train, X_test, Y_train, Y_test

def test_cpu(X_test, Y_test):
    test = Dataset(X_test, Y_test)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    model = torch.load('models/hybrid.pth')

    i = 0
    with torch.no_grad():
        for train_input, train_label in tqdm(test_dataloader):
            mask = train_input['attention_mask']
            input_id = train_input['input_ids'].squeeze(1)
            output = model(input_id, mask)
            if i==0:
                predict_Y_test = output.argmax(dim=1).cpu().numpy()
                i=1
            else:
                predict_Y_test = np.append(predict_Y_test, output.argmax(dim=1).cpu().numpy())

    Y_test.replace({'Label': {'ham': 0, 'spam': 1}}, inplace=True)
    Y_test = Y_test['Label']
    P = precision_score(Y_test, predict_Y_test, average='micro')
    acc = accuracy_score(Y_test, predict_Y_test)
    R = recall_score(Y_test, predict_Y_test, average='micro')
    F1 = f1_score(Y_test, predict_Y_test, average='micro')

    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, predict_Y_test), display_labels=['Spam', 'Ham'])
    display.plot()
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(Y_test, predict_Y_test)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('Hybrid_Model: acc =', acc, ' P =', P, ' R =', R, ' F1 =', F1)

def test(X_test, Y_test):
    test = Dataset(X_test, Y_test)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    model = torch.load('models/hybrid.pth')

    i = 0
    with torch.no_grad():
        for train_input, train_label in tqdm(test_dataloader):
            mask = train_input['attention_mask'].to('cuda')
            input_id = train_input['input_ids'].squeeze(1).to('cuda')
            output = model(input_id, mask)
            if i==0:
                predict_Y_test = output.argmax(dim=1).cpu().numpy()
                i=1
            else:
                predict_Y_test = np.append(predict_Y_test, output.argmax(dim=1).cpu().numpy())

    Y_test.replace({'Label': {'ham': 0, 'spam': 1}}, inplace=True)
    Y_test = Y_test['Label']
    P = precision_score(Y_test, predict_Y_test, average='micro')
    acc = accuracy_score(Y_test, predict_Y_test)
    R = recall_score(Y_test, predict_Y_test, average='micro')
    F1 = f1_score(Y_test, predict_Y_test, average='micro')

    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, predict_Y_test), display_labels=['Spam', 'Ham'])
    display.plot()
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(Y_test, predict_Y_test)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('Hybrid_Model: acc =', acc, ' P =', P, ' R =', R, ' F1 =', F1)

def predict(x='Ya very nice. . .be ready on thursday'):
    model = torch.load('models/hybrid.pth', map_location=torch.device('cpu'))
    with torch.no_grad():
        X = tokenizer(x,
                   padding='max_length',
                   max_length=512,
                   truncation=True,
                   return_tensors="pt")
        mask = X['attention_mask'].to('cpu')
        input_id = X['input_ids'].squeeze(1).to('cpu')
        output = model(input_id, mask)
        y = output.argmax(dim=1).cpu().numpy()

    return y[0]

def train(model, X_train, X_test, Y_train, Y_test, learning_rate, epochs):
    # Get training and validation sets from Dataset classes
    train, val = Dataset(X_train, Y_train), Dataset(X_test, Y_test)
    # DataLoader gets the data according to batch_size, and chooses to scramble the sample during training
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # Determine whether to use GPU
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Define loss functions and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # Start the training cycle
    for epoch_num in range(epochs):
        # Define two variables to store the accuracy and loss of the training set
        total_acc_train = 0
        total_loss_train = 0
        # Progress bar function tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = torch.Tensor(train_label).long()
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # Get output through the model
            output = model(input_id, mask)
            # Calculate the loss
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # Calculation accuracy
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # Model update
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ Verify the model -----------
        # Define two variables that store the accuracy and loss of the validation set
        total_acc_val = 0
        total_loss_val = 0
        # No need to calculate the gradient
        with torch.no_grad():
            # Loop the data set and validate it with the trained model
            for val_input, val_label in val_dataloader:
                # If there is a GPU, then use the GPU, the next operation is the same as training
                val_label = torch.Tensor(val_label).long()
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(X_train): .3f} 
              | Train Accuracy: {total_acc_train / len(X_train): .3f} 
              | Val Loss: {total_loss_val / len(X_test): .3f} 
              | Val Accuracy: {total_acc_val / len(X_test): .3f}''')
    torch.save(model, 'models/hybrid.pth')

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = get_data()
    EPOCHS = 5
    LR = 1e-6
    model = Hybrid()
    train(model, X_train, X_test, Y_train, Y_test, LR, EPOCHS)
    test(X_test, Y_test)
    # print(predict())