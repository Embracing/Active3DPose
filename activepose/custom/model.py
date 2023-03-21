from functools import partial

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'MultiLayerPerceptron',
    'MLP',
    'MixtureDensityNetwork',
    'MDN',
    'SelfAttention',
]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_dims, activation=nn.ReLU(), output_activation=None):
        super().__init__()

        self.activation = activation
        self.output_activation = output_activation

        self.linear_layers = nn.ModuleList()
        for i in range(len(n_dims) - 1):
            self.linear_layers.append(
                module=nn.Linear(in_features=n_dims[i], out_features=n_dims[i + 1], bias=True)
            )

        self.in_features = n_dims[0]
        self.out_features = n_dims[-1]

    def forward(self, x):
        n_layers = len(self.linear_layers)
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < n_layers - 1:
                x = self.activation(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


MLP = MultiLayerPerceptron


def hook(grad, name=None):
    if not torch.isfinite(grad.data).all():
        grad_no_nan = torch.nan_to_num(grad, nan=0.0)
        if name is not None:
            print(name, grad, grad_no_nan.max(), grad_no_nan.min())
        else:
            print(grad, grad_no_nan.max(), grad_no_nan.min())


class MixtureDensityNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_hidden_dims,
        n_gaussians,
        activation=nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_gaussians = n_gaussians
        self.raw_out_features = raw_out_feature = n_gaussians * (1 + 2 * out_features)

        self.mlp = MultiLayerPerceptron(
            n_dims=[in_features, *n_hidden_dims, raw_out_feature], activation=activation
        )

        for name, param in self.mlp.named_parameters():
            param.register_hook(partial(hook, name=name))

    def forward(self, x, epsilon=None):
        out = self.mlp(x)

        pi_logits = out[..., : self.n_gaussians]
        pi_logits = pi_logits - pi_logits.mean(dim=-1, keepdims=True)
        pi_logits = pi_logits.clamp(min=-1e8, max=+1e8)
        pi = F.softmax(pi_logits, dim=-1)
        mu = out[..., self.n_gaussians : self.n_gaussians * (1 + self.out_features)]
        log_sigma = out[
            ...,
            self.n_gaussians
            * (1 + self.out_features) : self.n_gaussians
            * (1 + 2 * self.out_features),
        ]
        log_sigma = log_sigma.clamp(min=-20, max=+20)
        sigma = torch.exp(log_sigma)

        if epsilon is None:
            epsilon = torch.finfo(sigma.dtype).eps
        sigma = sigma.clamp_min(epsilon)

        out = torch.cat([pi_logits, mu, sigma], dim=-1)

        mu = mu.view(*mu.size()[:-1], self.n_gaussians, self.out_features)
        sigma = sigma.view(*sigma.size()[:-1], self.n_gaussians, self.out_features)

        if sigma.requires_grad:
            sigma.register_hook(partial(hook, name='sigma'))

        assert not torch.isnan(pi).any()
        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()

        return pi, mu, sigma, out

    def distribution(self, x, epsilon=None):
        pi, mu, sigma, out = self(x, epsilon=epsilon)
        return MixtureDensity(pi, mu, sigma, out)

    def log_prob(self, x, y, epsilon=None):
        distribution = self.distribution(x, epsilon=epsilon)
        return distribution.log_prob(y)

    @torch.no_grad()
    def sample(self, x, sample_shape=torch.Size([]), epsilon=None):
        distribution = self.distribution(x, epsilon=epsilon)
        return distribution.sample(sample_shape=sample_shape)


class MixtureDensity:
    def __init__(self, pi, mu, sigma, out):
        self.pi = pi  # (*, G)
        self.mu = mu  # (*, G, D_out)
        self.sigma = sigma  # (*, G, D_out)
        self.out = out

    def log_prob(self, y):
        y = y.unsqueeze(dim=-2).to(self.sigma.dtype)
        y = y - self.mu
        edge = (5 * self.sigma).detach()
        y = torch.where(y > edge, edge, y)
        y = torch.where(y < -edge, -edge, y)
        comp_log_prob = D.Normal(0.0, self.sigma).log_prob(y).sum(dim=-1)
        max_comp_log_prob = comp_log_prob.max(dim=-1)[0]
        comp_log_prob = comp_log_prob - max_comp_log_prob.unsqueeze(dim=-1)
        comp_prob = torch.exp(comp_log_prob)
        prob = (self.pi * comp_prob).sum(dim=-1)
        prob = prob.clamp(min=1e-20, max=1e20)
        log_prob = torch.log(prob)
        log_prob = max_comp_log_prob + log_prob
        return log_prob, self.out

    @property
    def mean(self):
        pi = self.pi.unsqueeze(dim=-1)
        mean = pi * self.mu
        mean = mean.sum(dim=-2)
        return mean

    @property
    def variance(self):
        pi = self.pi.unsqueeze(dim=-1)
        variance = pi * self.sigma.pow(2)
        variance = variance.sum(dim=-2)
        return variance

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size([])):
        index = D.OneHotCategorical(probs=self.pi).sample(sample_shape=sample_shape).bool()
        distribution = D.Normal(self.mu, self.sigma)
        z = distribution.rsample(sample_shape=sample_shape)
        return z[index]


MDN = MixtureDensityNetwork


class SelfAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, activation=F.relu, batch_first=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.activation = activation

        self.self_attn = nn.MultiheadAttention(
            self.in_features, num_heads=num_heads, bias=True, batch_first=batch_first
        )

        self.linear1 = nn.Linear(in_features, in_features, bias=True)
        self.linear2 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x, w = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.activation(self.linear1(x))
        return self.linear2(x), w
