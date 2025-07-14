# Not useful. It's just for me to test the model

from recommind_packages import Processor, NeuMF, trainer
import torch
import os
from dotenv import load_dotenv
load_dotenv()  
models_path = os.getenv('models')

ncf_path = os.path.join(models_path, "ncf_model")



recommind_model = torch.load(os.path.join(ncf_path, 'recommind_model.pth'))
model = NeuMF(**recommind_model['config'])
model.load_state_dict(recommind_model['model_state_dict'])


model.train()
number_batch = 0
losses = []
y_true = []
y_pred = []
for X, y in trainloader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()