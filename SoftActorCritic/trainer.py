class SACTrainer:
    def __init__(self, env, agent, replay_buffer, batch_size=256,
                 start_steps=1000, update_after=1000, update_every=50,
                 max_episode_steps=200):
        """
        env: OpenAI Gym environment.
        agent: SACAgent instance.
        replay_buffer: ReplayBuffer instance.
        batch_size: Batch size for training.
        start_steps: Number of initial steps with random actions.
        update_after: Minimum steps before updates start.
        update_every: Frequency (in steps) to perform training updates.
        max_episode_steps: Maximum steps per episode.
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.max_episode_steps = max_episode_steps

    def run(self, num_episodes=100):
        total_steps = 0
        for episode in range(num_episodes):
            state, _ = self.env.reset()

            episode_reward = 0
            for t in range(self.max_episode_steps):
                if total_steps < self.start_steps:
                    # Use random actions to encourage exploration initially.
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward
                total_steps += 1

                if total_steps >= self.update_after and total_steps % self.update_every == 0:
                    for _ in range(self.update_every):
                        losses = self.agent.update(self.replay_buffer, self.batch_size)
                    # Optionally print losses:
                    # print(losses)

                if done:
                    break

            print(f"Episode: {episode+1:03d} | Reward: {episode_reward:.2f}")