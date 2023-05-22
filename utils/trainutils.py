import torch

import numpy as np

def train(model, optimizer, train_loader, loss_fn, epoch, report_freq=100):
    model.train(True)
    totalcorrect = 0
    for iteration, (x, y) in enumerate(train_loader):
        x = x.to('cuda')
        y = y.to('cuda')
        optimizer.zero_grad()
        y_hat = model(x, None) #! add another parameter instead of hardcoding None
        loss = loss_fn(y_hat, y)
        prediction = torch.round(torch.sigmoid(y_hat))
        correct = prediction.eq(y.view_as(prediction)).sum().item()
        totalcorrect += correct
        loss.backward()
        optimizer.step()
        if iteration % report_freq == 0:
            print('Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}'.format(epoch, iteration, 
                                                            loss.item(), correct / len(y)))
    print('Epoch: {}, Total Train Accuracy: {}'.format(epoch, totalcorrect / len(train_loader.dataset)))
    return model


def test(model, test_loader, loss_fn, epoch):
    model.train(False)
    loss = 0
    correct = 0
    pred_y = []
    count = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            y_hat = model(x, None) #! add another parameter instead of hardcoding None
            pred_y.append(y_hat)
            loss += loss_fn(y_hat, y).item()
            prediction = torch.round(torch.sigmoid(y_hat))
            correct += prediction.eq(y.view_as(prediction)).sum().item()
            count += 1

    loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    pred_y = torch.cat(pred_y, dim=0)
    pred_y = torch.round(torch.sigmoid(pred_y))

    print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, accuracy))

    return loss, accuracy, pred_y

