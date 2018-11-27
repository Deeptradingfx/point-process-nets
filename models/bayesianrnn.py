import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    This recurrent neural network will learn the decay parameter of
    a hawkes process and generate a sequence of events.
    """
    def __init__(self, input_size, hidden_size, output_size,
                 mu, alpha, betas, p_betas):
        """
        betas (array): candidate values for the beta parameter
        p_betas (array): prior probability distribution of the beta values
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.mu = mu
        self.alpha = alpha
        self.betas = betas
        self.beta_priors = p_betas
        self.beta_posts = p_betas
        self.n_betas = self.betas.size(0)
        self.event_x = torch.zeros(input_size, dtype=torch.float64)
        # Intensity of each process in the batch
        self.intens = torch.empty(
            self.n_betas,
            dtype=torch.float64).fill_(self.mu)
        self.trajectories = torch.zeros(1, self.n_betas, dtype=torch.float64)

        print("mu: %s" % self.mu)
        print(self.intens.shape)
        print(self.trajectories.shape)

    def forward(self, events: torch.Tensor, hidden: torch.Tensor):
        # print("events.shape: %s" % str(events.shape))
        # print("hidden.shape: %s" % str(hidden.shape))
        # new times
        dt = -1.0/self.intens * \
            torch.rand(self.n_betas, dtype=torch.float64).log()
        prev = self.trajectories[-1]
        output = prev + dt

        self.trajectories = torch.cat(
            (self.trajectories, output.view(1, -1)), dim=0)
        maxlambda = self.intens.clone()
        self.intens = self.mu + (-self.betas*dt).exp()*(self.intens-self.mu) +\
            self.alpha*self.betas
        # check for new events happening
        u = torch.rand(self.n_betas, dtype=torch.float64)
        self.event_x = (u < self.intens/maxlambda).to(dtype=torch.float64)
        self.intens += self.alpha*self.betas*self.event_x

        # Update the posterior probabilities
        likel = self.likelihood(dt, self.event_x)
        self.beta_posts = likel*self.beta_posts
        self.beta_posts = self.beta_posts/self.beta_posts.sum()
        # print("Intensities: %s" % self.intens)
        # print("Output shape: %s" % str(output.shape))
        return output, hidden

    def likelihood(self, dt, x):
        """
        Compute the likelihoods of the next event happening within dt.
        """
        exponent = self.mu*dt + (self.intens - self.mu) *\
            (1 - (-self.betas*dt).exp())/self.betas
        fact = self.mu + (-self.betas*dt).exp()*(self.intens - self.mu)
        # return resulting likelihood
        return (1 - (-exponent).exp())*fact.pow(x)

    def initHidden(self):
        return torch.zeros(self.hidden_size, dtype=torch.float64)
