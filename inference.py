from __future__ import print_function
import torch
import numpy as np
from flask import Flask, request
import json
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from flask import request, jsonify, send_file
from PIL import Image
import io
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import base64
from flask import Flask, redirect, url_for, render_template, request, flash


model = None
app = Flask(__name__)

# Loading and transforming the dataset
test = datasets.MNIST("", train=False, download=True,
                  transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        #Softmax gets probabilities.
        return F.log_softmax(x, dim=1)

def load():
    global model
    model = Net()
    model.load_state_dict(torch.load("/mnist/vol/mnist_wt.pth"))
    model.eval()

@app.route('/ping', methods=['GET'])
def get_ping():
    res = "Ping successful"
    return res

@app.route('/load', methods=['GET'])
def load_model():
    load()
    return "MODEL SUCCESFULLY LOADED"

@app.route('/')
def login():
    return render_template('front.html')

@app.route('/img/<img>')
def get_image(img):
    return send_file("/mnist/images/"+img, mimetype='image/gif')

@app.route('/guess/', methods=['POST', 'GET'])
def get_data():
    imgid = request.form['Number']
    num = 1
    res = {}
    for data in testset:
        if num >= 10:
            break
        X, y = data
        if y.item() == int(imgid):
            digit = X
            fname2 = "num" + imgid + "_" + str(num) + ".png"
            fname = "/mnist/images/"+fname2
            plt.imsave(fname=fname, arr=digit.view(28,28), cmap='gray_r', format='png')

            # model classification
            input_string = json.dumps(digit.tolist())
            data_string = input_string
            data_array = json.loads(data_string)
            data_tensor = torch.Tensor(data_array)
            guess = torch.argmax(model(data_tensor)[0])

            # res
            key = "Prediction for Image " + str(num)
            res_string = "Classified as " + str(guess.item())
            res[key] = {"title": res_string, "img": fname2}
            num+=1
    return render_template('backend.html', result=res)

if __name__ == '__main__':
    load()
    app.run(host='0.0.0.0', port=9000)