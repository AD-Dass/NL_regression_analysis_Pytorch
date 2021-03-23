import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import imageio
torch.manual_seed(1)

data_frame = pd.read_csv("C:\\Users\\user\\Documents\\Projects\\NL_NN\\NL_Sample.csv")

# Determine the training and testing dataset and sizes
dataset_length = len(data_frame)
print("This the length of the provided dataset: ", dataset_length)
train_len = dataset_length*80//100
trainframe = data_frame.iloc[0:train_len,:]
testframe = data_frame.iloc[train_len:,:]
test_len=len(testframe)

#Create a dataset class to contain and handle our data

class Data():
    def __init__(self,train):
        if train == True:
            self.y=torch.tensor(trainframe.iloc[:,1].values,dtype=torch.float).reshape((-1,1))
            self.x=torch.tensor(trainframe.iloc[:,0].values, dtype=torch.float).reshape((-1,1))
            self.len=self.x.shape[0]
            print("Train dataset length: ",self.len)
        else:
            self.y=torch.tensor(testframe.iloc[:,1].values,dtype=torch.float).reshape((-1,1))
            self.x=torch.tensor(testframe.iloc[:,0].values, dtype=torch.float).reshape((-1,1))
            self.len=self.x.shape[0]
            print("Test dataset length: ",self.len)
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


#Call our train and test data
care_train = Data(train=True)
care_test = Data(train=False)


#HyperParameter tune controls
batches=50
learnin_rate = 0.01 #0.004
epochs = 150 #102
print("Network is being trained under with:")
print("Epochs: ", epochs)
print("Learning Rate: ", learnin_rate)
print("Batches: ", batches)


#put our datasets in to dataloader format
train_loader = DataLoader(dataset=care_train,batch_size=batches,shuffle=True)
test_loader = DataLoader(dataset=care_test,batch_size=1,shuffle=True)

#Design neural net. In this instance we are using a two layer network 1-x-1
class NET(nn.Module):
    def __init__(self,inputs=0,hidden_nodes=0,outputs=0):
        super(NET,self).__init__()
        self.hidden = nn.Linear(inputs,hidden_nodes)
        self.linear = nn.Linear(hidden_nodes,outputs)

    def forward(self,x):
        x = torch.nn.functional.leaky_relu(self.hidden(x)) #using leaky relu activition function to ensure that neuron updates are enacted and avoiding deadpsots that occur in the relu
        yhat=self.linear(x)
        return yhat
    
#Estbalish Model
model=NET(inputs=1,hidden_nodes=200,outputs=1)

#Using Stochastic Gradient descent 
optimizer = optim.SGD(model.parameters(), lr = learnin_rate) #0.02-0.005
#using Mean squared error loss
criterion = nn.MSELoss()
#print(model.state_dict())

#Just a reference to show model output
x_rep=[-2.5,-2.0,-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
test_look = torch.FloatTensor(x_rep).reshape((-1,1))  

#Some lists to store some useful information to use later
my_images =[]
LOSS = []
Accuracy = []

#For our animated plot
fig,ax=plt.subplots(figsize=(12,7))

#Now for training and testing our Nueral Network
for epoch in range(epochs):
    model.train() #sets the model into training mode
    for x,y in train_loader:
        yhat = model(x)
        loss = criterion(yhat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        LOSS.append(loss.item())

    model.eval() #sets our model into evaluation mode for testing and visualisaiton
    for x_test, y_test in test_loader:
        yhat = model(x_test)
        loss_f = ((y_test-yhat)/y_test)*100
        Accuracy.append(loss_f.item())

    #Lets create our gif to visualise the training of the Nueral network
    results = model(test_look)
    results = results.reshape((1,-1))
    results = results.tolist()
    results=results[0]
    plt.cla()
    ax.set_title('Training an AI for Regression Analysis',fontsize=26)
    ax.set_ylim(-5.0,30.0)
    ax.set_xlim(-5.0,5.0)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_xlabel("X",fontsize=16)
    ax.scatter(data_frame['X'],data_frame['Y_M'],color='tab:orange')
    ax.plot(x_rep,results)
    ax.text(-4.0,2.0,'Step = %d' % epoch,fontdict={'size':16, 'color': 'red'})
    ax.text(-4.0,0.0,'Loss = %.3f' % loss.item(),fontdict={'size':16, 'color': 'red'})

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1]+(3,))

    my_images.append(image)

imageio.mimsave('./curve_1.gif', my_images, fps=10)

    

#plot the loss
headarg_1 = 'ML Learning Cost (LR/BS/E = '
headarg_2 = str(learnin_rate)
headarg_3 = str(batches)
headarg_4 = str(epochs)

color = 'tab:red'
plt.cla()
plt.plot(LOSS,color=color)
plt.xlabel('Epoch',color=color)
plt.ylabel('Cost',color=color)
plt.title(headarg_1 + headarg_2+ '/' + headarg_3 + '/' +headarg_4+')')
#plt.ylim(bottom=0,top=100)
plt.show()

color = 'tab:blue'
plt.plot(Accuracy,color=color)
plt.xlabel('Epoch',color=color)
plt.ylabel('Accurary (%)',color=color)
#plt.ylim(bottom=-150,top=150)
plt.title('Model Validation Accuracy')
plt.show()
