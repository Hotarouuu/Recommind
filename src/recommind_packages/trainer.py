import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import StepLR

def trainer(model, path, trainloader, evalloader, epochs = 40, device = 'cuda', early_stopping = True, n_factors = 16, lr = 0.0005, weight_decay = 1e-5):

    num_epochs = epochs
    device = device

    model.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


    if early_stopping:

        # Early Stopping Parameters
        patience = 10          
        min_delta = 0.001    
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False

        for it in range(num_epochs):
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

                number_batch += 1

                if number_batch % 250 == 0:
                    
                    print(f'Batch atual: {number_batch}, Batch_Loss: {loss.item()}')
            
            print(f'\nIter #{it}', f'Loss: {sum(losses)/len(losses)}, LearningRate: {optimizer.param_groups[0]["lr"]}\n')
            print(f'\nEvaluating...')

            model.eval()
            val_loss = 0.0

            for X, y in evalloader:
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    outputs = model(X)
                    eval_loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
                    val_loss += eval_loss.item() * X.size(0)


                    y_pred.append(outputs.squeeze().cpu().numpy())

                    y_true.append(y.cpu().numpy())
            val_loss /= len(evalloader.dataset) 


            mse = mean_squared_error(y_true, y_pred)

            rmse = np.sqrt(mse)
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f'Evaluation Loss: {val_loss}\n')

            
            # Early Stopping
            if it >= 15:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
            
                    torch.save(model.state_dict(), os.path.join(path, 'best_model.pth'))
                    print(f'    --> New best eval loss: {best_val_loss:.4f}. Model saved.')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'Early stopping activated in epoch {it+1}.')
                        early_stop = True
                        break

            if early_stop:
                break
            scheduler.step()

    else: 

        for it in range(num_epochs):
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

                number_batch += 1

                if number_batch % 250 == 0:
                    
                    print(f'Batch atual: {number_batch}, Batch_Loss: {loss.item()}')
            
            print(f'\nIter #{it}', f'Loss: {sum(losses)/len(losses)}, LearningRate: {optimizer.param_groups[0]["lr"]}\n')
            print(f'\nEvaluating...')

            model.eval()
            val_loss = 0.0

            for X, y in evalloader:
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    outputs = model(X)
                    eval_loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
                    val_loss += eval_loss.item() * X.size(0)


                    y_pred.append(outputs.squeeze().cpu().numpy())

                    y_true.append(y.cpu().numpy())
            val_loss /= len(evalloader.dataset) 


            mse = mean_squared_error(y_true, y_pred)

            rmse = np.sqrt(mse)
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f'Evaluation Loss: {val_loss}\n')
