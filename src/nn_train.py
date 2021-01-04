import sys
import warnings

warnings.filterwarnings('ignore')

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from my_dataloader import My_DataLoader
from lstm import *
from pytorch_tool import EarlyStopping

country = 'Russia'
if len(sys.argv) > 1:
    country = sys.argv[1]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", sys.argv[2])

lr = 0.01
epochs = 10000
batch_size = 36
verbose = True
verify_every = 1
patience = 5
use_cuda = True
predict_gradiant = True
normalization = True
mae_loss = True
grad_norm = False
grad_norm_alpha = 0.12

loss_weights = [0.3, 0.3, 0.3, 0, 0.1, 0.1, 0]
loss_weights = torch.from_numpy(np.array(loss_weights)).to(torch.float).cuda()

dl = My_DataLoader()
dl.load_from_csv()

ori_data = dl.countries_data[country].drop(columns=['date'])
ori_norm_data = dl.norm_countries_data[country]
valid_start = ori_norm_data[ori_norm_data['date'] == '11/10/20'].index.values[0]
valid_end = ori_norm_data[ori_norm_data['date'] == '12/10/20'].index.values[0]
ori_norm_data = ori_norm_data.drop(columns=['date'])
feature_size = ori_norm_data.shape[1]
param_data = dl.norm_param_countries_data[country].drop(columns=['date']).values

checkpoint_path = dl.root_path + 'model/checkpoint_{}.pt'.format(country)
result_path = dl.root_path + 'result/'

early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=checkpoint_path)

config = LSTM_Config(feature_size=ori_norm_data.shape[1], hidden_size=256, time_steps=100)

index_map = {}
for i, col in enumerate(ori_data.columns.values):
    index_map[col] = i

x_window_train = []
for i in range(config.time_steps, valid_start):
    # print('i', i)
    new_x = []
    for j in range(i - config.time_steps, i):
        # print('j', j)
        if normalization:
            row = ori_norm_data.loc[j].values.tolist()
        else:
            row = ori_data.drop(columns=['day_of_the_week']).loc[j].values.tolist()

        new_x.append(row)

    x_window_train.append(new_x)
    # print(df.shape)
if predict_gradiant:
    y_train = ori_data[config.time_steps: valid_start][['new_cases', 'new_deaths', 'new_recovered', 'new_tests']]
else:
    y_train = ori_data[config.time_steps: valid_start][['total_cases', 'total_deaths', 'total_recovered', 'total_tests']]

if normalization:
    x_valid = ori_norm_data[valid_start - config.time_steps: valid_end].to_numpy()
else:
    x_valid = ori_data.drop(columns=['day_of_the_week'])[valid_start - config.time_steps: valid_end].to_numpy()

if predict_gradiant:
    y_valid_data = ori_data[valid_start: valid_end][['new_cases', 'new_deaths', 'new_recovered', 'new_tests']]
else:
    y_valid_data = ori_data[valid_start: valid_end][['total_cases', 'total_deaths', 'total_recovered', 'total_tests']]

y_valid = torch.from_numpy(y_valid_data.values.astype('float')).to(torch.float).cuda()

train_data = TensorDataset(torch.from_numpy(np.array(x_window_train)).float(),
                           torch.from_numpy(y_train.values.astype('float')).float())
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)

#
# train_data = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
# valid_data = TensorDataset(torch.from_numpy(x_valid.values), torch.from_numpy(y_valid.values))
# test_data = TensorDataset(torch.from_numpy(data_loader.x_test.values), torch.from_numpy(data_loader.y_test.values))
# train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
# valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)
# test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
#
# data = data_loader.data
# feature_sizes = []
# for col in data.columns:
#     if col in data_loader.continuous_column:
#         feature_sizes.append(1)
#     elif col == 'is_trade':
#         continue
#     else:
#         feature_sizes.append(len(data[col].unique()) + 50)
#
# net = DeepFM(feature_sizes=feature_sizes, continuous_field_size=len(data_loader.continuous_column), num_classes=2,
#              use_cuda=use_cuda)
# optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.03)
#
# if use_cuda:
#     net.cuda()
# model = net.fit(train_loader, valid_loader, optimizer, epochs=epochs, verbose=verbose, print_every=print_every)
#
# test_loss = net.validation(test_loader, model)
# print('test_log_loss: {0:f}'.format(test_loss))
#


net = LSTM_Model(config)
# net = nn.DataParallel(net, device_ids=[0, 2])
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs * 0.6, epochs * 0.8], gamma=0.9)

criterion_valid = nn.MSELoss(reduction='mean').cuda()


class My_Train_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor, info):
        # print(type(output), type(target), type(info))
        # print(output.shape, target.shape, info.shape)

        losses = []

        this_batch_size = output.shape[0]

        for i in range(4):
            dis = output[:, i] - target[:, i]
            losses.append(dis)

        if predict_gradiant:
            # dA/dt - alpha D
            losses.append(output[:, 1] - info[:this_batch_size, index_map['death_rate']] *
                          info[:this_batch_size, index_map['current_patients']])
            # dR/dt - gamma D
            losses.append(output[:, 2] - info[:this_batch_size, index_map['recovery_rate']] *
                          info[:this_batch_size, index_map['current_patients']])

            # dD/dt - beta T
            losses.append(
                output[:, 0] - info[:this_batch_size, index_map['positive_rate']] * info[1: 1 + this_batch_size,
                                                                                    index_map['new_cases']])

        else:
            # dA/dt - alpha Dƒ
            losses.append(
                output[:, 1] - info[:this_batch_size, index_map['total_deaths']] - info[:this_batch_size,
                                                                                   index_map['death_rate']] * info[
                                                                                                              :this_batch_size,
                                                                                                              index_map[
                                                                                                                  'current_patients']])
            # dR/dt - gamma D
            losses.append(
                output[:, 2] - info[:this_batch_size, index_map['total_recovered']] - info[:this_batch_size,
                                                                                      index_map[
                                                                                          'recovery_rate']] * info[
                                                                                                              :this_batch_size,
                                                                                                              index_map[
                                                                                                                  'current_patients']])
            # dD/dt - beta T
            losses.append(
                output[:, 0] - info[:this_batch_size, index_map['total_cases']] - info[:this_batch_size,
                                                                                  index_map['positive_rate']] * (
                    info[1: 1 + this_batch_size, index_map['new_cases']]))

        for i in range(7):
            if mae_loss:
                losses[i] = torch.abs(losses[i]).mean()
            else:
                losses[i] = (losses[i] * losses[i]).mean()

        loss = torch.stack(losses).cuda()
        loss = loss * loss_weights

        return loss


criterion_train = My_Train_Loss().cuda()
# criterion_train = nn.MSELoss(reduction='mean').cuda()

task_losses = []
loss_ratios = []
grad_norm_losses = []
weights = []

train_losses = []
valid_losses = []
avg_train_losses = []
avg_task_losses = []
avg_valid_losses = []
for epoch in range(1, epochs + 1):
    net.train()
    for batch, (x, y_target) in enumerate(train_loader, 1):
        x = x.cuda()
        y_target = y_target.cuda()
        start_index = batch * batch_size + config.time_steps - 1
        info = torch.from_numpy(ori_data[start_index: start_index + batch_size + 1].values).to(torch.float).cuda()

        y_list = net(x)
        y = torch.stack(y_list).reshape((-1, 4))
        task_loss = criterion_train(y, y_target, info)
        if grad_norm:
            # compute the weighted loss w_i(t) * L_i(t)
            weighted_task_loss = torch.mul(net.weights, task_loss)
            if epoch == 1:
                # initialize the initial loss L(0) if t=0
                initial_task_loss = task_loss.data.cpu().numpy()

            # get the total loss
            loss = torch.sum(weighted_task_loss)
            # clear the gradients
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            # This is equivalent to compute each \nabla_W L_i(t)
            loss.backward(retain_graph=True)

            task_losses.append(task_loss.data.cpu().numpy())
            train_losses.append(task_loss.data.cpu().numpy().sum())

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            net.weights.grad.data = net.weights.grad.data * 0.0

            # get layer of shared weights
            W = net.fc[0]

            # get the gradient norms for each of the tasks
            # G^{(i)}_w(t)
            norms = []
            for i in range(len(task_loss)):
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                # compute the norm
                norms.append(torch.norm(torch.mul(net.weights[i], gygw[0])))
            norms = torch.stack(norms)

            # compute the inverse training rate r_i(t)
            # \curl{L}_i
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)

            # compute the mean norm \tilde{G}_w(t)
            mean_norm = np.mean(norms.data.cpu().numpy())

            # compute the GradNorm loss
            # this term has to remain constant
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** grad_norm_alpha), requires_grad=False)
            constant_term = constant_term.cuda()

            # this is the GradNorm loss itself
            grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
            # compute the gradient for the weights
            net.weights.grad = torch.autograd.grad(grad_norm_loss, net.weights)[0]


        else:
            loss = task_loss.sum()
            optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.cpu().item())

        optimizer.step()

    # renormalize
    if grad_norm:
        normalize_coeff = config.n_tasks / torch.sum(net.weights.data, dim=0)
        net.weights.data = net.weights.data * normalize_coeff
        grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())

        avg_task_loss = np.mean(task_losses, axis=0)
        avg_task_losses.append(avg_task_loss)
        loss_ratios.append(np.sum(avg_task_losses[-1] / avg_task_losses[0]))
        weights.append(net.weights.data.cpu().numpy())

    loss_train = np.average(np.array(train_losses))
    avg_train_losses.append(loss_train)

    print(country, 'epoch', epoch, "loss", loss_train)
    if grad_norm:
        print('task loss', avg_task_loss)
        print('task weights', weights[-1])

    scheduler.step()
    train_losses = []
    task_losses = []

    if epoch % verify_every == 0:
        net.eval()
        # y_predict_list = []
        y_predict = None
        for i in range(valid_start, valid_end):
            x_window_valid = x_valid[i - valid_start: i - valid_start + config.time_steps].reshape(1, config.time_steps, -1)
            x_window_valid = torch.from_numpy(x_window_valid.astype('float')).to(torch.float).cuda()
            t_y_predict = net(x_window_valid)
            t_y_predict = torch.stack(t_y_predict).reshape((-1, 4))
            if y_predict is None:
                y_predict = t_y_predict
            else:
                y_predict = torch.cat([y_predict, t_y_predict], dim=0)

            y_np = t_y_predict.cpu().detach().numpy().reshape((-1))

            x_np = np.zeros(feature_size)

            if predict_gradiant:
                x_np[1] = y_np[0]
                x_np[3] = y_np[1]
                x_np[5] = y_np[2]
                x_np[7] = y_np[3]
                x_np[0] = ori_data.loc[i - 1, 'total_cases'] + x_np[1]
                x_np[2] = ori_data.loc[i - 1, 'total_deaths'] + x_np[3]
                x_np[4] = ori_data.loc[i - 1, 'total_recovered'] + x_np[5]
                x_np[6] = ori_data.loc[i - 1, 'total_tests'] + x_np[7]

            else:
                x_np[0] = y_np[0]
                x_np[2] = y_np[1]
                x_np[4] = y_np[2]
                x_np[6] = y_np[3]
                x_np[1] = x_np[0] - ori_data.loc[i - 1, 'total_cases']
                x_np[3] = x_np[2] - ori_data.loc[i - 1, 'total_deaths']
                x_np[5] = x_np[4] - ori_data.loc[i - 1, 'total_recovered']
                x_np[7] = x_np[6] - ori_data.loc[i - 1, 'total_tests']

            x_np[8] = x_np[3] / ori_data.loc[i - 1, 'current_patients']
            x_np[9] = x_np[5] / ori_data.loc[i - 1, 'current_patients']
            x_np[10] = x_np[1] / x_np[7]
            x_np[11] = x_np[0] - x_np[2] - x_np[4]

            if normalization:
                for j in range(feature_size - 2):
                    x_np[j] = (x_np[j] - param_data[0, j]) / param_data[1, j]

            day_of_the_week = (ori_data.loc[i - 1, 'day_of_the_week'] + 1) % 7
            x_np[12] = np.sin(2 * np.pi * day_of_the_week / 6.0)
            x_np[13] = np.cos(2 * np.pi * day_of_the_week / 6.0)

            # x_valid = np.row_stack([x_valid, x_np])
            x_valid[i - valid_start] = x_np

        loss_valid = criterion_valid(y_predict[:, :3], y_valid[:, :3]).cpu().item()

        avg_valid_losses.append([epoch, loss_valid])

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {loss_train:.5f} ' +
                     f'valid_loss: {loss_valid:.5f}')

        print(print_msg)

        early_stopping(loss_valid, net)

        if early_stopping.early_stop:
            print('Early stooping!')
            break

        result = y_predict.cpu().detach().numpy()
        if predict_gradiant:
            s = ori_data[['total_cases', 'total_deaths', 'total_recovered', 'total_tests']][
                valid_start - 1: valid_start].values
            for r in range(result.shape[0]):
                s += result[r]
                result[r] = s

        np.savetxt('{}result_{}.csv'.format(result_path, country), result, delimiter=',')
        np.savetxt('{}train_loss_{}.csv'.format(result_path, country), np.array(avg_train_losses), delimiter=',')
        np.savetxt('{}valid_loss_{}.csv'.format(result_path, country), np.array(avg_valid_losses), delimiter=',')

        if grad_norm:
            np.savetxt('{}loss_weights_{}.csv'.format(result_path, country), np.array(weights), delimiter=',')
            np.savetxt('{}task_loss_{}.csv'.format(result_path, country), np.array(avg_task_loss), delimiter=',')
            np.savetxt('{}loss_ratios_{}.csv'.format(result_path, country), np.array(loss_ratios), delimiter=',')


net.load_state_dict(torch.load(early_stopping.path))
net.eval()
y_predict = None
for i in range(valid_start, valid_end):
    x_window_valid = x_valid[i - valid_start: i - valid_start + config.time_steps].reshape(1, config.time_steps, -1)
    x_window_valid = torch.from_numpy(x_window_valid.astype('float')).to(torch.float).cuda()
    t_y_predict = net(x_window_valid)
    t_y_predict = torch.stack(t_y_predict).reshape((-1, 4))
    if y_predict is None:
        y_predict = t_y_predict
    else:
        y_predict = torch.cat([y_predict, t_y_predict], dim=0)

    y_np = t_y_predict.cpu().detach().numpy().reshape((-1))

    x_np = np.zeros(feature_size)

    if predict_gradiant:
        x_np[1] = y_np[0]
        x_np[3] = y_np[1]
        x_np[5] = y_np[2]
        x_np[7] = y_np[3]
        x_np[0] = ori_data.loc[i - 1, 'total_cases'] + x_np[1]
        x_np[2] = ori_data.loc[i - 1, 'total_deaths'] + x_np[3]
        x_np[4] = ori_data.loc[i - 1, 'total_recovered'] + x_np[5]
        x_np[6] = ori_data.loc[i - 1, 'total_tests'] + x_np[7]

    else:
        x_np[0] = y_np[0]
        x_np[2] = y_np[1]
        x_np[4] = y_np[2]
        x_np[6] = y_np[3]
        x_np[1] = x_np[0] - ori_data.loc[i - 1, 'total_cases']
        x_np[3] = x_np[2] - ori_data.loc[i - 1, 'total_deaths']
        x_np[5] = x_np[4] - ori_data.loc[i - 1, 'total_recovered']
        x_np[7] = x_np[6] - ori_data.loc[i - 1, 'total_tests']

    x_np[8] = x_np[3] / ori_data.loc[i - 1, 'current_patients']
    x_np[9] = x_np[5] / ori_data.loc[i - 1, 'current_patients']
    x_np[10] = x_np[1] / x_np[7]
    x_np[11] = x_np[0] - x_np[2] - x_np[4]

    if normalization:
        for j in range(feature_size - 2):
            x_np[j] = (x_np[j] - param_data[0, j]) / param_data[1, j]

    day_of_the_week = (ori_data.loc[i - 1, 'day_of_the_week'] + 1) % 7
    x_np[12] = np.sin(2 * np.pi * day_of_the_week / 6.0)
    x_np[13] = np.cos(2 * np.pi * day_of_the_week / 6.0)

    # x_valid = np.row_stack([x_valid, x_np])
    x_valid[i - valid_start] = x_np

loss_valid = criterion_valid(y_predict[:, :3], y_valid[:, :3]).cpu().item()
result = y_predict.cpu().detach().numpy()
if predict_gradiant:
    s = ori_data[['total_cases', 'total_deaths', 'total_recovered', 'total_tests']][valid_start - 1: valid_start].values
for r in range(result.shape[0]):
    s += result[r]
    result[r] = s

np.savetxt('{}result_{}_at_{}_loss_{:.3f}.csv'.format(result_path, country, epoch - patience, loss_valid), result,
           delimiter=',')
