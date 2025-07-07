import torch
import torch.nn as nn
import torch.nn.functional as F

def dayid(i):
    return str(i) if i >= 10 else '0' + str(i)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.sum()


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation is None:
        return nn.Identity()
    else:
        raise ValueError("this activation function has not been specified in 'get_activation' yet")


class DNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, dropout_rate=0.0, use_bn=False):
        super(DNN_layer, self).__init__()
        dnn_layer = [nn.Linear(input_dim, output_dim)]
        if use_bn:
            dnn_layer.append(nn.BatchNorm1d(output_dim))
        dnn_layer.append(get_activation(activation))
        if dropout_rate > 0:
            dnn_layer.append(nn.Dropout(dropout_rate))
        self.dnn_layer = nn.Sequential(*dnn_layer)

    def forward(self, inputs):
        return self.dnn_layer(inputs)


class MultiLayerDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation=None, dropout=0.0, use_bn=False):
        super(MultiLayerDNN, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(
                DNN_layer(
                    input_dim=dims[i],
                    output_dim=dims[i+1],
                    activation=activation,
                    dropout_rate=dropout,
                    use_bn=use_bn
                )
            )
        if output_dim:
            layers.append(
                DNN_layer(input_dim=dims[-1], output_dim=output_dim)
            )
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)



class GateNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, dropout_rate=0.0, use_bn=False):
        super(GateNN_layer, self).__init__()
        gate_layers = [nn.Linear(input_dim, output_dim)]
        if use_bn:
            gate_layers.append(nn.BatchNorm1d(output_dim))
        gate_layers.append(get_activation(activation))
        if dropout_rate > 0:
            gate_layers.append(nn.Dropout(dropout_rate))
        gate_layers.append(nn.Linear(output_dim, output_dim))
        gate_layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, inputs):
        return 2 * self.gate(inputs)


class PPNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation=None, dropout=0.0, use_bn=False):
        super(PPNetBlock, self).__init__()
        mlp_layers, gate_layers = [], []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            mlp_layers.append(
                DNN_layer(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    activation=activation,
                    dropout_rate=dropout,
                    use_bn=use_bn
                )
            )
            gate_layers.append(
                GateNN_layer(
                    input_dim=input_dim,
                    output_dim=dims[i],
                    activation=activation,
                    dropout_rate=dropout,
                    use_bn=use_bn
                )
            )
        if output_dim:
            mlp_layers.append(
                DNN_layer(input_dim=dims[-1], output_dim=output_dim)
            )
            gate_layers.append(
                GateNN_layer(input_dim=dims[-1], output_dim=output_dim)
            )
        self.mlp_layers = nn.Sequential(*mlp_layers)
        self.gate_layers = nn.Sequential(*gate_layers)


    def forward(self, other_embs, id_embs):
        gate_input = torch.cat([other_embs.detach(), id_embs], dim=-1)
        all_input = torch.cat([other_embs, id_embs], dim=-1)
        x = all_input
        for i in range(len(self.mlp_layers)):
            gw = self.gate_layers[i](gate_input)
            x = self.mlp_layers[i](x * gw)
        return x


