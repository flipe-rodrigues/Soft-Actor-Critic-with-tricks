import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, 
                 activation=nn.ReLU, output_activation=nn.Identity):
        """
        Build a fully connected neural network with a configurable number
        of hidden layers and units per layer.
        
        input_dim: Dimension of input.
        output_dim: Dimension of output.
        hidden_layers: List of integers, each representing the size of a hidden layer.
        activation: Activation function for hidden layers.
        output_activation: Activation function for the output layer.
        """
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation=nn.ReLU):
        """
        Policy network outputs both mean and log_std for a Gaussian distribution,
        then samples an action using the reparameterization trick.
        """
        super(PolicyNetwork, self).__init__()
        # Output: concatenated mean and log_std for each action dimension.
        self.net = MLP(state_dim, action_dim * 2, hidden_layers, activation=activation)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Returns:
            action: Sampled action after applying tanh squashing.
            log_prob: Log probability of the action (with tanh correction).
            mean: The mean output by the network.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # Sample using reparameterization.
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # Compute log probability (with correction for tanh squashing).
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + 1e-6), dim=-1, keepdim=True)
        action = y_t
        return action, log_prob, mean

    def get_action(self, state, deterministic=False):
        # Move state tensor to the same device as the network parameters.
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.sigmoid(mean)
        else:
            action, _, _ = self.sample(state)
        return action.detach().cpu().numpy()[0]



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation=nn.ReLU):
        """
        Q-network takes state and action as input and outputs a scalar Q-value.
        """
        super(QNetwork, self).__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_layers, activation=activation)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

