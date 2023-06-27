from time import sleep
from controller.cnn import CNN
from controller.dqn_agent import DQNAgent
from controller.epsilon_profile import EpsilonProfile
from game.SpaceInvaders import SpaceInvaders


def main():

    game = SpaceInvaders(display=True)

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    gamma = 1.
    n_episodes = 2000
    max_steps = 500
    alpha = 0.001
    eps_profile = EpsilonProfile(1.0, 0.1)
    final_exploration_episode = 5
    batch_size = 32
    replay_memory_size = 1000
    target_update_frequency = 100
    tau = 1.0

    model = CNN(game.nx, game.ny, game.na)

    print('--- neural network ---')
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print([param.numel() for param in model.parameters() if param.requires_grad])
    print('number of parameters:', num_params)
    print(model)

    controller = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode, game.na)

    controller.learn(game, n_episodes, max_steps)

    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        #sleep(0.0001)


if __name__ == '__main__':
    main()
