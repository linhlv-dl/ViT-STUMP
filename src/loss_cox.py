
# code from: https://github.com/KatherLab/marugoto/blob/survival/marugoto/survival/loss.py
import torch

def cox_loss(y_true, y_pred):
    time_value = torch.squeeze(y_true[0:, 0])
    event = torch.squeeze(y_true[0:, 1]).type(torch.bool)
    score = torch.squeeze(y_pred)

    ix = torch.where(event)[0]

    sel_time = time_value[ix]
    sel_mat = (sel_time.unsqueeze(1).expand(1, sel_time.size()[0],
                                            time_value.size()[0]).squeeze() <= time_value).float()

    p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score), axis=-1))

    loss = -torch.mean(p_lik)

    return loss

def cox_loss_custom(y_true_time, y_true_event, y_pred):
    time_value = torch.squeeze(y_true_time)
    #print(time_value.size())
    event = torch.squeeze(y_true_event).type(torch.bool)
    score = torch.squeeze(y_pred)

    ix = torch.where(event)[0]

    sel_time = time_value[ix]
    sel_mat = (sel_time.unsqueeze(1).expand(1, sel_time.size()[0],
                                            time_value.size()[0]).squeeze() <= time_value).float()

    p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score), axis=-1))

    loss = -torch.mean(p_lik)

    return loss


def concordance_index(time_value, event, y_pred):
    time_value = time_value.squeeze()
    event = event.squeeze()
    y_pred = y_pred.squeeze()
    #time_value = torch.squeeze(y_true[0:, 0])
    #event = torch.squeeze(y_true[0:, 1]).type(torch.bool)

    time_1 = time_value.unsqueeze(1).expand(1, time_value.size()[0], time_value.size()[0]).squeeze()
    event_1 = event.unsqueeze(1).expand(1, event.size()[0], event.size()[0]).squeeze()
    ix = torch.where(torch.logical_and(time_1 < time_value, event_1))

    s1 = y_pred[ix[0]]
    s2 = y_pred[ix[1]]
    ci = torch.mean((s1 < s2).float())
    return ci

