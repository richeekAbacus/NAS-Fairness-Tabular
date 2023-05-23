import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(dataset_train, dataset_val, dataset_test, train_bs, test_bs,
                    num_workers=1, pin_memory=True):
    train_x, train_y = torch.Tensor(dataset_train.features), torch.Tensor(dataset_train.labels)
    val_x, val_y = torch.Tensor(dataset_val.features), torch.Tensor(dataset_val.labels)
    test_x, test_y = torch.Tensor(dataset_test.features), torch.Tensor(dataset_test.labels)    
    
    train_loader = DataLoader(dataset=TensorDataset(train_x, train_y), batch_size=train_bs,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=TensorDataset(val_x, val_y), batch_size=test_bs,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=TensorDataset(test_x, test_y), batch_size=test_bs,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

