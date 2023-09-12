# From: https://colab.research.google.com/github/shrey920/MultivariateTimeSeriesForecasting/blob/master/MultivariateTimeSeriesForecasting(colab_version).ipynb#scrollTo=Rspn1MTDZYCj
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot
from torch.autograd import Variable
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import dash
import dash_core_components as dcc
import dash_html_components as html


class LSTNet(nn.Module):
    def __init__(self, args, data):
        super(LSTNet, self).__init__()
        #         self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # autoregressive
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res

# @title
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, dataframe, train, valid, horizon, window, normalize=0):
        #         self.cuda = cuda;
        self.P = window
        self.h = horizon
        self.rawdat = dataframe
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        # print(self.test[0].size(), len(self.test), self.test[1].size())
        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        #         if self.cuda:
        #             self.scale = self.scale.cuda();
        #         self.scale = Variable(self.scale);

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        if (normalize == 0):
            # scalers = {}
            for i in self.rawdat.columns:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                s_s = scaler.fit_transform(self.rawdat[i].values.reshape(-1, 1))
                s_s = np.reshape(s_s, len(s_s))
                # scalers['scaler_' + i] = scaler
                self.rawdat[i] = s_s
            self.dat = self.rawdat.to_numpy()

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))
        if (normalize == 3):
            self.dat = self.rawdat.to_numpy()

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        # Y = torch.zeros((n,1))
        # print('X Y sefl.dat shapes:')
        # print(X.shape, Y.shape, self.dat.shape)
        # print('n and idx_set:')
        # print(n, idx_set)
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
            # Y[i] = torch.from_numpy(np.asarray(self.dat[idx_set[i],1]))
            # print('start, end ', start, end)
            # print('i, ind_set[i]', i, idx_set[i])
            # print(' X[i].shape, y[i].shape', X[i].shape, Y[i].shape)
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            #             if (self.cuda):
            #                 X = X.cuda()
            #                 Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size


def evaluate2(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        # print(X.shape)
        # print(X)
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data
        n_samples += (output.size(0) * data.m)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = 0
    # correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    # correlation = (correlation[index]).mean()
    return rse, rae, correlation, predict, Ytest


class Arguments():
    def __init__(self, data, hidCNN=100, hidRNN=100, window=35, CNN_kernel=6, highway_window=24, clip=10, epochs=5,
                 batch_size=128, dropout=0.2, save="save.pt", optim="adam", lr=0.001, horizon=1, skip=24, hidSkip=5,
                 L1loss=True, normalize=0, output_fun="sigmoid", port = '8050'):
        self.data = data
        self.hidCNN = hidCNN
        self.hidRNN = hidRNN
        self.window = window
        self.CNN_kernel = CNN_kernel
        self.highway_window = highway_window
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.optim = optim
        self.lr = lr
        self.skip = skip
        self.normalize = normalize
        self.horizon = horizon
        self.save = save
        self.output_fun = output_fun
        self.hidSkip = hidSkip
        self.L1Loss = L1loss,
        self.port = port



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        # Objective Function
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm


def load_dataset(training_file, testing_file):
    dataframes = []
    for data_file in [training_file, testing_file]:
        parser = lambda data_string: datetime.strptime(data_string, '%Y-%m-%d %H:%M:%S')
        dataframe = pd.read_csv(data_file, parse_dates=['Time'],
                                date_parser=parser, index_col=0)
        print(f"Rows in {data_file}: {len(dataframe)}")
        dataframe.drop(['Ls', 'LT', 'CO2ice'], axis=1, inplace=True)
        dataframe.index.name = "Time"

        dataframes.append(dataframe)

    return pd.concat(dataframes, axis=0)


dataframe = load_dataset('data/data_files/insight_openmars_training_time.csv',
                         'data/data_files/insight_openmars_test_time.csv')

args = Arguments(horizon=12, skip = 24, window = 35, hidCNN=30, hidRNN=30, L1loss=False, data=dataframe, save=f'model_files/LSTNET/lstnet_model_hor_12_win_100_skip_24.pt', output_fun=None,
                 normalize=3, epochs=10, port = '8054')
# args.save = 'model_files/LSTNET/lstnet_model.pt'
print('Model selected - ', args.save)
with open(args.save, 'rb') as f:
    model = torch.load(f)

Data = Data_utility(args.data, 0.8, 0.1, args.horizon, args.window, args.normalize)

nParams = sum([p.nelement() for p in model.parameters()])

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)

# best_val = 100  # 1000000;
# print(Data.test[0])
# print(Data.test[1])

test_acc, test_rae, test_corr, predict, Ytest = evaluate2(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                          evaluateL1, args.batch_size)

columns = ['Tsurf', 'Psurf', 'cloud', 'vapour', 'u_wind', 'v_wind', 'dust', 'temp']
test_df = pd.DataFrame(Ytest, columns=columns, index=dataframe[-8856:].index)
predict_df = pd.DataFrame(predict, columns=columns, index=dataframe[-8856:].index)

# test_df.to_csv('test_data.csv')
# predict_df.to_csv('predicted_data.csv')

def multi_plot(test_df, predict_df, title):
    default_column = 'Tsurf'
    all_columns = []
    fig = go.Figure()
    for column in test_df.columns.to_list():
        # add line / trace 1 to figure
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df[column],
            marker=dict(
                color="blue"
            ),
            name='Actual',
            visible=(column == default_column)

        ))

        # add line / trace 2 to figure
        fig.add_trace(go.Scatter(
            x=predict_df.index,
            y=predict_df[column],
            marker=dict(
                color="red"
            ),
            name='Predicted',
            visible=(column == default_column)
        ))
        all_columns.extend([column] * 2)

    buttons = []
    for column in test_df.columns.to_list():
        # create a button object for the column-pair
        button = dict(label=column,
                      method="update",
                      args=[{"visible": [column == c for c in all_columns]}])

        # add the button to our list of buttons
        buttons.append(button)

    fig.update_layout(showlegend=True,
                      updatemenus=[{
                          'active': 0,
                          'buttons': buttons,

                          'showactive': True
                          # yaxis_type="log"
                      }
                      ]
                      )
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        height=800

    )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         step="month",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )

        # fig.show()
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(debug=True, use_reloader=False, port = int(args.port))  # Turn off reloader if inside Jupyter

multi_plot(test_df, predict_df, title=f"Comparing actual and predicted values of 8 variables from OpenMars data. Model: {args.save}")

