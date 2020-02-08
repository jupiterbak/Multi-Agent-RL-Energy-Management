import os
import numpy as np
import matplotlib.pyplot as plt


DATA_ROOT_PATH = 'C:/GitHub/Multi-Agent-RL-Energy-Management/Visualisation/experiment_2'
EXPORT_ROOT_PATH = 'C:/GitHub/Multi-Agent-RL-Energy-Management/Visualisation/export'
SEEDS = [998, 79, 784, 667, 2148]  # [998, 667, 11, 2148, 79]
SCENARIO_COUNT = 1
AGENT_COUNT = 4
DATA_SIZE = 300
AGENT_NAMES = [
    ["Welding Machine", "CNC Machine", "Energy Storage", "Wind Turbine"],
    ["Machine 1", "Machine 2", "Storage", "Generator"],
    ["Machine 1", "Machine 2", "Storage", "Generator"]
]


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
        scenario_reward['all'] = np.array(cul_reward)[:, :DATA_SIZE]

        # Get energy prices
        tous = []
        for seed in SEEDS:
            _path_tous = os.path.normpath('{}/scenario_{}/{}/run-.-tag-current_energy_price.csv'
                                         .format(data_path, scenario, seed))
            tous.append(np.loadtxt(_path_tous, skiprows=1, delimiter=','))
        scenario_reward['tou_prices'] = np.array(tous)[:, :DATA_SIZE]

        # Get current power
        power = []
        for seed in SEEDS:
            _path_power = os.path.normpath('{}/scenario_{}/{}/run-.-tag-current_system_power.csv'
                                         .format(data_path, scenario, seed))
            power.append(np.loadtxt(_path_power, skiprows=1, delimiter=','))
        scenario_reward['power'] = np.array(power)[:, :DATA_SIZE]

        # Get load profile
        load = []
        for seed in SEEDS:
            _path_load = os.path.normpath('{}/scenario_{}/{}/run-.-tag-max_load_pofile.csv'
                                         .format(data_path, scenario, seed))
            load.append(np.loadtxt(_path_load, skiprows=1, delimiter=','))
        scenario_reward['load'] = np.array(load)[:, :DATA_SIZE]

        # Get production
        load = []
        for seed in SEEDS:
            _path_production = os.path.normpath('{}/scenario_{}/{}/run-.-tag-production.csv'
                                          .format(data_path, scenario, seed))
            load.append(np.loadtxt(_path_production, skiprows=1, delimiter=','))
        scenario_reward['production'] = np.array(load)[:, :DATA_SIZE]

        # Get all agent rewards
        for agent in range(AGENT_COUNT):
            agent_reward = []
            for seed in SEEDS:
                _path_agent = os.path.normpath('{}/scenario_{}/{}/run-.-tag-cul_reward_agent_{}.csv'
                                               .format(data_path, scenario, seed, agent))
                agent_reward.append(np.loadtxt(_path_agent, skiprows=1, delimiter=','))
            scenario_reward['agent_{}'.format(agent)] = np.array(agent_reward)[:, :DATA_SIZE]

        rslt['scenario_{}'.format(scenario)] = scenario_reward
    return rslt


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.2):
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


def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.2):
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


def plot_all_learning_curve(subplot, train_sizes, train_scores, title, alpha=0.2):
    # These are the colors that will be used in the plot
    # subplot.set_prop_cycle(color=[
    #     '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    #     '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    #     '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    #     '#17becf', '#9edae5'])
    # These are the colors that will be used in the plot
    # subplot.set_prop_cycle(color=[
    #     '#d62728', '#9467bd', '#8c564b',
    #     '#e377c2', '#7f7f7f', '#bcbd22',
    #     '#17becf'])

    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)

    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)

    subplot.plot(train_sizes, train_mean,  color='tab:blue', label="Global Reward")
    subplot.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, color='tab:blue', alpha=alpha)

    # subplot.set_title(title, fontsize='12', fontweight='bold')
    subplot.set_xlabel('Episodes', fontsize='9')
    subplot.set_ylabel('Global Reward', fontsize='9')
    subplot.grid(ls='--')
    # subplot.legend(loc='best')
    # subplot.legend(frameon=True, fontsize=8)


def plot_agent_learning_curve(subplot, train_sizes, train_scores, title, alpha=0.2):

    # These are the colors that will be used in the plot
    subplot.set_prop_cycle(color=[
        '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22',
        '#17becf'])

    # Remove the plot frame lines. They are unnecessary here.
    # subplot.spines['top'].set_visible(False)
    # subplot.spines['bottom'].set_visible(False)
    # subplot.spines['right'].set_visible(False)
    # subplot.spines['left'].set_visible(False)
    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)

    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)

    subplot.plot(train_sizes, train_mean,  label=title)
    subplot.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, alpha=alpha)

    # subplot.set_title(title, fontsize='9')
    subplot.set_xlabel('Episodes', fontsize='9')
    subplot.set_ylabel('Reward', fontsize='9')
    subplot.grid(ls='--')
    # subplot.legend(loc='best')
    subplot.legend(frameon=True, fontsize=8)


def plot_agents_learning_curve(subplot, agent_train_data_sizes, agent_train_data_scores, titles=None, alpha=0.2):
    # These are the colors that will be used in the plot
    subplot.set_prop_cycle(color=[
        '#9467bd', '#ff7f0e', '#2ca02c',
        '#d62728', '#bcbd22', '#8c564b',
        '#e377c2', '#7f7f7f', '#17becf'])

    # Remove the plot frame lines. They are unnecessary here.
    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)

    for a in range(AGENT_COUNT):
        train_mean = np.mean(agent_train_data_scores[a], axis=0)
        train_std = np.std(agent_train_data_scores[a], axis=0)
        _label = titles[a] if not titles is None else 'agent_{}'.format(a)
        subplot.plot(agent_train_data_sizes[a], train_mean, label=_label)
        subplot.fill_between(agent_train_data_sizes[a], train_mean + train_std,
                             train_mean - train_std, alpha=alpha)

    # subplot.set_title(title, fontsize='9')
    subplot.set_xlabel('Episodes', fontsize='9')
    subplot.set_ylabel('Reward', fontsize='9')
    subplot.grid(ls='--')
    subplot.legend(loc='best')
    subplot.legend(frameon=True, fontsize=8)


def plot_plant_performance(subplot, production__sizes, production_performance, _power_consumtions, alpha=0.7):
    # Prepare the data
    production_performance_mean = np.mean(production_performance, axis=0)
    power_consumptions_mean = np.mean(_power_consumtions, axis=0)

    ax2 = subplot.twinx()  # instantiate a second axes that shares the same x-axis

    # Remove the plot frame lines. They are unnecessary here.
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax2.grid(ls='--')
    ax2.grid(False)
    ax2.plot(production__sizes, production_performance_mean, linewidth=2,
                 color='tab:green', label="Executed Production Tasks")
    ax2.set_xlabel('Episodes', fontsize='9')
    ax2.set_ylabel('Executed Production Tasks', fontsize='9')
    ax2.legend(loc='best')
    ax2.legend(frameon=True, fontsize=8)
    # ax2.set_title("TOU price ($/kWh)", fontsize='9')

    # ax2.legend(loc='best', frameon=True, fontsize=8)
    subplot.plot(production__sizes, power_consumptions_mean, color='tab:blue', label="Current Power")
    subplot.fill_between(production__sizes, power_consumptions_mean, 0, color='tab:blue', alpha=alpha)
    subplot.set_ylabel('Power (kWh)', fontsize='9')
    subplot.set_xlabel('Episodes', fontsize='9')
    # subplot.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    subplot.tick_params(axis='y', rotation=0)
    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)


def plot_energy_costs(subplot, train_sizes, energy_prices, power_consumptions, alpha=0.7):
    # Prepare the data
    energy_prices_mean = np.mean(energy_prices, axis=0)
    power_consumptions_mean = np.mean(power_consumptions, axis=0)
    energy_cost_mean = np.multiply(energy_prices_mean, power_consumptions_mean)

    ax2 = subplot.twinx()  # instantiate a second axes that shares the same x-axis

    subplot.plot(train_sizes, energy_cost_mean, color='tab:blue', label="Energy Cost ($/h)")
    subplot.fill_between(train_sizes, energy_cost_mean, 0, color='tab:blue', alpha=alpha)
    # subplot.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    subplot.tick_params(axis='y', rotation=0)
    subplot.xaxis.set_tick_params(labelsize=8)
    subplot.yaxis.set_tick_params(labelsize=8)
    subplot.set_ylabel('Energy Cost ($/h)', fontsize='9')
    # subplot.legend(loc='best')
    # subplot.legend(frameon=True, fontsize=8)

    # Remove the plot frame lines. They are unnecessary here.
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax2.plot(train_sizes, energy_prices_mean,  '-.', linewidth=2,
                 color='tab:brown', label="TOU price ($/kWh)")

    # ax2.set_title("TOU price ($/kWh)", fontsize='9')
    ax2.set_xlabel('Episodes', fontsize='9')
    ax2.set_ylabel('TOU price ($/kWh)', fontsize='9')
    ax2.grid(ls='--')
    ax2.legend(loc='best')
    ax2.legend(frameon=True, fontsize=8)
    ax2.grid(False)


if __name__ == '__main__':
    raw_data = parse_raw_data(DATA_ROOT_PATH)

    # Configure the Plot
    plt.style.use('ggplot')

    # fig, axes = plt.subplots(3, SCENARIO_COUNT, sharex=True, figsize=(20, 9))
    fig, axes = plt.subplots(3, SCENARIO_COUNT, figsize=(10, 8))

    scenario_data = raw_data['scenario_{}'.format(0)]

    # Global Reward
    train_sizes = np.arange(scenario_data['all'].shape[1])
    train_scores = scenario_data['all'][:, :, 2]

    plot_all_learning_curve(axes[0], train_sizes=train_sizes, train_scores=train_scores,
                            title="Scenario {}".format(0), alpha=0.3)

    # Agent rewards
    a_data_sizes = []
    a_data_scores = []
    for agent in range(AGENT_COUNT):
        agent_train_sizes = np.arange(scenario_data['agent_{}'.format(agent)].shape[1])
        agent_train_scores = scenario_data['agent_{}'.format(agent)][:, :, 2]

        a_data_sizes.append(agent_train_sizes)
        a_data_scores.append(agent_train_scores)

    plot_agents_learning_curve(axes[1], agent_train_data_sizes=a_data_sizes,
                               agent_train_data_scores=a_data_scores, titles=AGENT_NAMES[0], alpha=0.3)

    # Production and Energy Costs
    production_sizes = np.arange(scenario_data['production'].shape[1])
    production_performances = scenario_data['production'][:, :, 2]
    energy_prices_sizes = np.arange(scenario_data['tou_prices'].shape[1])
    energy_prices = scenario_data['tou_prices'][:, :, 2]
    power_sizes = np.arange(scenario_data['power'].shape[1])
    power_consumptions = scenario_data['power'][:, :, 2]

    # Production
    plot_plant_performance(axes[2], production__sizes=production_sizes,
                           production_performance=production_performances,
                           _power_consumtions=power_consumptions,
                           alpha=0.5)

    fig.tight_layout()
    plt.show()

    _save_path_pdf = os.path.normpath('{}/scenario_2_result.pdf'.format(EXPORT_ROOT_PATH))
    _save_path_svg = os.path.normpath('{}/scenario_2_result.svg'.format(EXPORT_ROOT_PATH))
    _save_path_png = os.path.normpath('{}/scenario_2_result.png'.format(EXPORT_ROOT_PATH))
    plt.savefig(_save_path_pdf, bbox_inches='tight')
    plt.savefig(_save_path_svg)
    plt.savefig(_save_path_png, dpi=300, transparent=True)
