# Implementation of the Soft-Actor-Critic method to solve continuous actions Reinforcement Learning
Based on the paper:

Soft Actor-Critic Algorithms and Applications
https://arxiv.org/abs/1812.05905
Also relevant:
Continuous control with deep reinforcement learning
https://arxiv.org/abs/1509.02971

List of tricks used:
- Twin Q-Networks
- Memory Replay (Experience Replay Buffer)
- Target Networks with Polyak Averaging
- Automatic Entropy Tuning
- Reparameterization Trick
- Tanh Action Squashing with Log-Probability Correction
- Gradient Clipping
- Random Action Initialization

