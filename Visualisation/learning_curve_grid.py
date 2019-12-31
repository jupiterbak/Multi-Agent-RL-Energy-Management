import os
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT_PATH = 'C:\GitHub\Multi-Agent-RL-Energy-Management\Visualisation\data'
SEEDS = [998, 667, 11, 2148, 79]
SCENARIO_COUNT = 4
AGENT_COUNT = 3


def parse_raw_data(data_path):
    rslt = {}
    for scenario in range(SCENARIO_COUNT):
        scenario_reward = {}
        # Get cul_all rewards
        cul_reward = []
        for seed in SEEDS:
            _path_all = os.path.normpath('{}/scenario_{}/{}/run-.-tag-cul_reward.csv'
                                         .format(data_path, scenario, seed))
            cul_reward.append(np.loadtxt(_path_all, skiprows=1, delimiter=','))
        scenario_reward['all'] = np.array(cul_reward)

        # Get all agent rewards
        for agent in range(AGENT_COUNT):
            agent_reward = []
            for seed in SEEDS:
                _path_agent = os.path.normpath('{}/scenario_{}/{}/run-.-tag-cul_reward_agent_{}.csv'
                                               .format(data_path, scenario, seed, agent))
                agent_reward.append(np.loadtxt(_path_agent, skiprows=1, delimiter=','))
            scenario_reward['agent_{}'.format(agent)] = np.array(agent_reward)

        rslt['scenario_{}'.format(scenario)] = scenario_reward
    return rslt


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('F-measure')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()


def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    param_range = [x[1] for x in param_range]
    sort_idx = np.argsort(param_range)
    param_range = np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Weight of class 2')
    plt.ylabel('Average values and standard deviation for F1-Score')
    plt.legend(loc='best')
    plt.show()


def plot_all_learning_curve(subplot, train_sizes, train_scores, title, alpha=0.1):
    # These are the colors that will be used in the plot
    subplot.set_prop_cycle(color=[
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5'])

    # Remove the plot frame lines. They are unnecessary here.
    subplot.spines['top'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    subplot.spines['left'].set_visible(False)
    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)

    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)

    subplot.plot(train_sizes, train_mean,  label=title)
    subplot.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, alpha=alpha)

    # subplot.set_title(title, fontsize='9')
    subplot.set_xlabel('Epochs', fontsize='9')
    subplot.set_ylabel('Reward', fontsize='9')
    subplot.grid(ls='--')
    # subplot.legend(loc='best')


def plot_agent_learning_curve(subplot, train_sizes, train_scores, title, alpha=0.1):

    # These are the colors that will be used in the plot
    subplot.set_prop_cycle(color=[
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5'])

    # Remove the plot frame lines. They are unnecessary here.
    subplot.spines['top'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    subplot.spines['left'].set_visible(False)
    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)

    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)

    subplot.plot(train_sizes, train_mean,  label=title)
    subplot.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, alpha=alpha)

    # subplot.set_title(title, fontsize='9')
    subplot.set_xlabel('Epochs', fontsize='9')
    subplot.set_ylabel('Reward', fontsize='9')
    subplot.grid(ls='--')
    # subplot.legend(loc='best')


def plot_agents_learning_curve(subplot, agent_train_data_sizes, agent_train_data_scores, alpha=0.1):
    for a in range(AGENT_COUNT):
        train_mean = np.mean(agent_train_data_scores[a], axis=0)
        train_std = np.std(agent_train_data_scores[a], axis=0)

        subplot.plot(agent_train_data_sizes[a], train_mean, label='agent_{}'.format(a))
        subplot.fill_between(agent_train_data_sizes[a], train_mean + train_std,
                             train_mean - train_std, alpha=alpha)

    subplot.grid(ls='--')
    subplot.legend(loc='best')


if __name__ == '__main__':
    raw_data = parse_raw_data(DATA_ROOT_PATH)

    # Configure the Plot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(AGENT_COUNT + 2, SCENARIO_COUNT, figsize=(16, 9))
    fig.tight_layout()

    for scenario in range(SCENARIO_COUNT):
        scenario_data = raw_data['scenario_{}'.format(scenario)]

        # Global Reward
        train_sizes = np.arange(scenario_data['all'].shape[1])
        train_scores = scenario_data['all'][:, :, 2]

        plot_all_learning_curve(axes[0, scenario], train_sizes=train_sizes, train_scores=train_scores,
                                title="Global Reward")

        # Agent rewards
        a_data_sizes = []
        a_data_scores = []
        for agent in range(AGENT_COUNT):
            agent_train_sizes = np.arange(scenario_data['agent_{}'.format(agent)].shape[1])
            agent_train_scores = scenario_data['agent_{}'.format(agent)][:, :, 2]
            plot_agent_learning_curve(axes[1 + agent, scenario], train_sizes=agent_train_sizes,
                                    train_scores=agent_train_scores,
                                    title="Agent {}".format(agent))

            # a_data_sizes.append(agent_train_sizes)
            # a_data_scores.append(agent_train_scores)
            # plot_agents_learning_curve(axes[(AGENT_COUNT + 2) * scenario + 1 + agent], agent_train_data_sizes=a_data_sizes,
            #                            agent_train_data_scores=a_data_scores)

        # Energy Costs
        plot_all_learning_curve(axes[1 + AGENT_COUNT, scenario], train_sizes=train_sizes, train_scores=train_scores,
                                title="Global Reward")

    plt.show()
