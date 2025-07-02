import torch

def test_model(device, model, test_loader, idx2class=None):
    y_pred_list = []
    y_true_list = []
    y_pred_prob=[]
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_prob.append(y_test_pred.cpu().numpy())
    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]

    return y_true_list, y_pred_list,y_pred_prob