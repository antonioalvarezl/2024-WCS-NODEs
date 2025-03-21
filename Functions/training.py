import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_batch):
        margin = 0.1
        sign_y = 1 - 2 * y_batch.float()
        loss = F.relu(sign_y * (y_pred - 0.5 + sign_y * margin))
        return loss.mean()

losses = {'mse': nn.MSELoss(), 
          'cross_entropy': nn.CrossEntropyLoss(),
          'bce': nn.BCEWithLogitsLoss(),
          'linear_sep': CustomLoss(),
         }

class  Trainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.
    """
    def __init__(self, model, optimizer, device, fixed_projector, loss_func = 'cross_entropy', AdjTrainer = False,
                 print_freq = 10, record_freq = 20, pathgifs = '', verbose = True):
        self.model, self.optimizer, self.device, self.loss_func = model, optimizer, device, loss_func
        self.loss = losses[loss_func]
        self.print_freq, self.record_freq, self.pathgifs, self.verbose= print_freq, record_freq, pathgifs, verbose
        self.adjtrainer, self.fixed_projector = AdjTrainer, fixed_projector
        
    def train(self, datatrain, max_epochs=30000, pathparams=''):
        # Initialize flags and parameters
        self.classif = self.noimp = self.relerr = self.nonconv = False
        self.max_epochs, self.patience = max_epochs, 10000
        self.best_score, self.best_epoch, self.model.best_param, last_saved_model = float('inf'), 0, 0., None
        self.histories = {'loss_history': [], 'acc_history': []}

        # Collect points and labels from datatrain
        points, labels = zip(*datatrain)
        points, labels = torch.cat(points), torch.cat(labels)

        # Set relative tolerance and consecutive iteration checks
        rel_tol, consecutive_iterations = 1e-10, 20
        
        print(f'Running until epoch={self.max_epochs}')
        
        for epoch in range(self.max_epochs):
            score, acc = self.run_epoch(datatrain)
            if self.verbose:
                print(f"Epoch {epoch + 1}. Avg loss: {score:.10f}.")
            self.histories['loss_history'].append(score)
            if not self.fixed_projector and self.loss_func=='cross_entropy':
                self.histories['acc_history'].append(acc)
            
            if score < self.best_score:
                self.best_score, self.best_epoch, self.model.best_param = score, epoch+1, self.model.state_dict()
                if last_saved_model is not None and os.path.exists(last_saved_model):
                    os.remove(last_saved_model)
                # Save the current model
                model_filename = pathparams + f'/best_param_NODE.pt'
                torch.save(self.model.state_dict(), model_filename)
                print('Saving model')
                last_saved_model, pat_epochs = model_filename, 0
            else:
                pat_epochs += 1
                if pat_epochs >= self.patience:
                    print(f'Stopping early due to no improvement for {self.patience} epochs')
                    self.noimp = True
                    break   
            if (epoch > 20000 and self.best_score > 0.15) or (epoch > 40000 and self.best_score > 0.1):
                print(f'Stopping early due to slow convergence')
                self.nonconv = True
                break
            if len(self.histories['loss_history']) > consecutive_iterations:
                # Calculate relative error
                rel_error = max(abs(self.histories['loss_history'][-1] - self.histories['loss_history'][-2-j]) / max(self.histories['loss_history'][-2-j], 1e-10) for j in range(consecutive_iterations))
                if rel_error < rel_tol:
                    print(f'Stopping early due to relative error {rel_error:.10f} < {rel_tol} for {consecutive_iterations} consecutive iterations')
                    self.relerr = True
                    break

            predictions, _ = self.model(points)
            condition_1, condition_0 = (labels == 1) & (predictions > 0.5), (labels == 0) & (predictions < 0.5)
            
            if (condition_0 | condition_1).all():
                print('Stopping condition met')
                self.trained, self.classif = True, True
                self.model.load_state_dict(self.model.best_param)
                return
        self.trained = True
        return
    
    def run_epoch(self, data):
        # Initialize variables
        epoch_loss, epoch_acc, data_len = 0., 0., len(data) 
        # Loop for training 
        for _, (x_batch, y_batch) in enumerate(data):
            if self.loss_func == 'mse' or self.loss_func == 'bce':
                y_batch = y_batch.float()
            self.optimizer.zero_grad()
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            y_pred, _ = self.model(x_batch)
            loss = self.loss(y_pred, y_batch)
            epoch_loss = loss.item()
            
            # Optimization step
            loss.backward()
            self.optimizer.step()
            if self.loss_func=='cross_entropy':
                m = nn.Softmax(dim=1)
                softpred = torch.argmax(m(y_pred), 1)
                accuracy=(softpred == y_batch).sum().item()/y_batch.size(0)
                epoch_acc += accuracy
        
        return epoch_loss / data_len, epoch_acc / data_len
        
