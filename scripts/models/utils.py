# Import packages
import numpy as np
import torch

# Helper functions used in various scripts and notebooks

def get_accuracy(preds, gts, class_dict):
    """
    Calculate accuracy (precision), weighted over classes
    preds: predicted labels
    gts: ground truth/target labels
    class_dict: dictionary of the shape {class_value (int): class_name (string)}
    returns: accuracy weighted over classes in percentage
    """
    mean_acc = 0
    for label in class_dict.keys():
        mean_acc += np.mean([pred == gt for pred, gt in zip(preds, gts) if gt == label])
    return mean_acc / len(class_dict) * 100


def eval_model(task, model, dataloader, device, class_dict):
    """
    Evaluate a MyGRU model on the given data.
    task: classification or regression
    model: MyGRU model to evaluate
    dataloader: torch.utils.data.Dataloader object containing data object data_utils.MyDataset
    device: device use for evaluation (for example cpu, cuda, ...)
    class_dict: dictionary of the shape {class_value (int): class_name (string)}
    returns: performance (accuracy in percentage for classification or RMSE for regression), the
    model predictions for all samples and the ground truth (target) values.
    """
    model.to(device)                        # Cast model to device
    model.eval()                            # Put model in evaluation mode
    preds = []                              # For tracking model predictions
    gts = []
    for _, data in enumerate(dataloader):   # Iterate through data batches in dataloader
        x, y = data                         # Unpack batch into samples and targets
        if task == 'classification':        # Cast targets to desired format
            y = y.long()
        else:
            y = y.float()
        x, y = x.to(device), y.to(device)   # Cast data to specified device

        output = model(x)                   # Get model predictions

        if task == 'classification':
            output = torch.argmax(output, dim=1)    # Get predicted labels from probabilities if classification

        preds.extend(output.cpu().detach().numpy()) # Store model predictions
        gts.extend(y.cpu().numpy())

    # Determine performance from predictions and target (ground truth) values
    if task == 'classification':
        performance = get_accuracy(preds, gts, class_dict)    
    else:
        performance = np.sqrt(np.mean((np.array(preds) - np.array(gts)) ** 2))
    return performance, preds, gts