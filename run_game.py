import sys
from time import sleep
from controller.cnn import CNN
from controller.dqn_agent import DQNAgent
from controller.epsilon_profile import EpsilonProfile
from controller.random_agent import RandomAgent
from game.SpaceInvaders import SpaceInvaders
import torch
import time

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(device)

def main():

    game = SpaceInvaders(display=True)

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    gamma = 1.
    n_episodes = 1500
    max_steps = 2000
    alpha = 0.001
    eps_profile = EpsilonProfile(1.0, 0.1)
    final_exploration_episode = 1000
    batch_size = 32
    replay_memory_size = 100
    target_update_frequency = 100
    tau = 1.0

    frame_skip_rate = 9

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = None

    controller = None

    if model_name == 'random':
        print('RandomAgent')
        controller = RandomAgent(4)
    elif model_name:
        print('loading model', model_name)
        model = torch.load(model_name, map_location=device)
    else:
        model = CNN(game.nx, game.ny, game.na)

    if not controller:
        print('--- neural network ---')
        num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print([param.numel() for param in model.parameters() if param.requires_grad])
        print('number of parameters:', num_params)
        print(model)

        controller = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode, game.na)

    if not model_name:
        print('learning')
        controller.learn(game, n_episodes, max_steps, frame_skip_rate)
        controller.epsilon = 0.0
        print('done learning')
        torch.save(controller.policy_net, 'model{}.pt'.format(str(time.time())))
        torch.save(controller.target_net, 'target{}.pt'.format(str(time.time())))

    # Test controller
    state = game.reset()
    is_done = False
    n_step_test = 5000
    total_reward = 0

    while not is_done and n_step_test > 0:
        action = controller.select_action(state)

        state, reward, is_done = game.step(action)
        #sleep(0.0001)
        n_step_test -= 1
        total_reward += reward

    if is_done:
        print("game over ! il restait :", n_step_test, "steps")
    print("reward:", total_reward)


if __name__ == '__main__':
    main()
