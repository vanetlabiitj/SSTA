import torch



def plain_train(model, x_natural, A_wave, edges, edge_weights,y, **kwargs):

    outs = model(x_natural, A_wave, edges, edge_weights)
    #print("shape of outs and y_truth", outs.shape, y.shape)
    #y_truth = y[..., 0]              # -> [512, 12, 207]
    #y_truth = y_truth.permute(0, 2, 1)  # -> [512, 207, 12]
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(outs, y)
    return loss
