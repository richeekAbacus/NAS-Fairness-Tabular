import rtdl
import torch

def train(model, optimizer, train_loader, loss_fn, epoch, report_freq=100):
    model.train(True)
    totalcorrect = 0
    for iteration, (x, y) in enumerate(train_loader):
        x = x.to('cuda')
        y = y.to('cuda')
        optimizer.zero_grad()
        if isinstance(model, rtdl.FTTransformer):
            y_hat = model(x, None) #! FTTransformer takes in x_num and x_cat
        else:
            y_hat = model(x)
        loss = loss_fn(y_hat, y)
        prediction = torch.round(torch.sigmoid(y_hat))
        correct = prediction.eq(y.view_as(prediction)).sum().item()
        totalcorrect += correct
        loss.backward()
        optimizer.step()
        if iteration % report_freq == 0:
            print('Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}'.format(epoch, iteration, 
                                                            loss.item(), correct / len(y)))


def test(model, test_loader, loss_fn, epoch):
    model.train(False)
    loss = 0
    correct = 0
    pred_y = []
    scores_y = []
    count = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            if isinstance(model, rtdl.FTTransformer):
                y_hat = model(x, None) #! add another parameter instead of hardcoding None
            else:
                y_hat = model(x)
            loss += loss_fn(y_hat, y).item()
            prediction = torch.round(torch.sigmoid(y_hat))
            scores_y.append(torch.sigmoid(y_hat))
            pred_y.append(prediction)
            correct += prediction.eq(y.view_as(prediction)).sum().item()
            count += 1

    loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    pred_y = torch.cat(pred_y, dim=0)
    scores_y = torch.cat(scores_y, dim=0)

    return loss, accuracy, pred_y, scores_y

