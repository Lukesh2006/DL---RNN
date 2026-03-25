# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement 

Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.
## Theory and Dataset
The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

<img width="761" height="817" alt="image" src="https://github.com/user-attachments/assets/d604f056-e948-4007-ab7f-b5c28693d069" />




## DESIGN STEPS
### STEP 1: Load and normalize data, create sequences.

### STEP 2: Convert data to tensors and set up DataLoader.

### STEP 3: Define the RNN model architecture.

### STEP 4: Summarize, compile with loss and optimizer.

### STEP 5: Train the model with loss tracking.

### STEP 6: Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: LUKESH M

### Register Number: 212224230144

```python
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc = nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

#-----hard code-----
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
      total_loss = 0
      for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss / len(train_loader))
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    # Plot training loss
    print('Name: LUKESH')
    print('Register Number: 212224230144')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
train_model(model,train_loader,criterion,optimizer)

```

### OUTPUT
<img width="1001" height="746" alt="image" src="https://github.com/user-attachments/assets/e03587f6-af44-442f-9a95-1882e3fc72a2" />


## True Stock Price, Predicted Stock Price vs time
<img width="1112" height="708" alt="image" src="https://github.com/user-attachments/assets/3c362f03-672b-41d9-92f7-201fbbd7ea49" />


### Predictions
<img width="636" height="53" alt="image" src="https://github.com/user-attachments/assets/d78b4577-06fb-4970-8d2f-82ed84f43aec" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
