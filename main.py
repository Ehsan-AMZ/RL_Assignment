from environment import ContextualEnvironment
# from policies import KLUCBSegmentPolicy, RandomPolicy, ExploreThenCommitSegmentPolicy, EpsilonGreedySegmentPolicy, TSSegmentPolicy, LinearTSPolicy
from policies import RandomPolicy, TSSegmentPolicy
import argparse
import json
import logging
import numpy as np
import pandas as pd
import time
import os.path


# List of implemented policies
def set_policies(policies_name, user_segment, user_segment_cascade, user_features, n_playlists):
    # Please see section 3.3 of RecSys paper for a description of policies
    POLICIES_SETTINGS = {
        'random' : RandomPolicy(n_playlists),
        'ts-seg-cascade' : TSSegmentPolicy(user_segment, user_segment_cascade, n_playlists, alpha_zero = 1, beta_zero = 1, model = 'cascade'),
        'ts-seg-DCM-beta1': TSSegmentPolicy(user_segment, user_segment_cascade, n_playlists, alpha_zero=1, beta_zero=1, model= 'DCM'),
        'ts-seg-DCM-beta30': TSSegmentPolicy(user_segment, user_segment_cascade, n_playlists, alpha_zero=1, beta_zero=30,
                                            model='DCM'),
        'ts-seg-DCM-beta99': TSSegmentPolicy(user_segment, user_segment_cascade, n_playlists, alpha_zero=1, beta_zero=99,
                                            model='DCM'),
    }

    return [POLICIES_SETTINGS[name] for name in policies_name]


if __name__ == "__main__":

    # Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--users_path", type = str, default = "data/user_features.csv", required = False,
                        help = "Path to user features file")
    parser.add_argument("--playlists_path", type = str, default = "data/playlist_features.csv", required = False,
                        help = "Path to playlist features file")
    parser.add_argument("--output_path", type = str, default = "results.json", required = False,
                        help = "Path to json file to save regret values")
    parser.add_argument("--policies", type = str, default = "random,ts-seg-cascade,ts-seg-DCM-beta1,ts-seg-DCM-beta30,ts-seg-DCM-beta99", required = False,
                        help = "Bandit algorithms to evaluate, separated by commas")
    parser.add_argument("--n_recos", type = int, default = 12, required = False,
                        help = "Number of slots L in the carousel i.e. number of recommendations to provide")
    parser.add_argument("--l_init", type = int, default = 3, required = False,
                        help = "Number of slots L_init initially visible in the carousel")
    parser.add_argument("--n_users_per_round", type = int, default = 10, required = False,
                        help = "Number of users randomly selected (with replacement) per round")
    parser.add_argument("--n_rounds", type = int, default = 100, required = False,
                        help = "Number of simulated rounds")
    parser.add_argument("--print_every", type = int, default = 10, required = False,
                        help = "Print cumulative regrets every 'print_every' round")

    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(__name__)

    if args.l_init > args.n_recos:
        raise ValueError('l_init is larger than n_recos')


    # Data Loading and Preprocessing steps

    logger.info("LOADING DATA")
    logger.info("Loading playlist data")
    playlists_df = pd.read_csv(args.playlists_path)

    logger.info("Loading user data\n \n")
    users_df = pd.read_csv(args.users_path)

    n_users = len(users_df)
    n_playlists = len(playlists_df)
    n_recos = args.n_recos
    print_every = args.print_every

    user_features = np.array(users_df.drop(["segment"], axis = 1))
    user_features = np.concatenate([user_features, np.ones((n_users,1))], axis = 1)
    playlist_features = np.array(playlists_df)

    user_segment = np.array(users_df.segment)
    user_segment_uniques = np.unique(user_segment)
    probability_no_cascade = 0
    random_temp = np.random.choice([0, 1], size=len(user_segment_uniques), p=[1 - probability_no_cascade, probability_no_cascade])
    user_segment_cascade = dict(zip(user_segment_uniques, random_temp))


    logger.info("SETTING UP SIMULATION ENVIRONMENT")
    logger.info("for %d users, %d playlists, %d recommendations per carousel \n \n" % (n_users, n_playlists, n_recos))

    cont_env = ContextualEnvironment(user_features, playlist_features, user_segment, user_segment_cascade, n_recos)

    logger.info("SETTING UP POLICIES")
    logger.info("Policies to evaluate: %s \n \n" % (args.policies))

    policies_name = args.policies.split(",")
    policies = set_policies(policies_name, user_segment, user_segment_cascade, user_features, n_playlists)
    n_policies = len(policies)
    n_users_per_round = args.n_users_per_round
    n_rounds = args.n_rounds
    overall_rewards = np.zeros((n_policies, n_rounds))
    overall_optimal_reward = np.zeros(n_rounds)


    # Simulations for Top-n_recos carousel-based playlist recommendations

    logger.info("STARTING SIMULATIONS")
    logger.info("for %d rounds, with %d users per round (randomly drawn with replacement)\n \n" % (n_rounds, n_users_per_round))
    start_time = time.time()
    for i in range(n_rounds):
        # Select batch of n_users_per_round users
        user_ids = np.random.choice(range(n_users), n_users_per_round)
        overall_optimal_reward[i] = np.take(cont_env.th_rewards, user_ids).sum()
        # Iterate over all policies
        for j in range(n_policies):
            # Compute n_recos recommendations
            recos = policies[j].recommend_to_users_batch(user_ids, args.n_recos, args.l_init)
            # Compute rewards
            rewards, probb = cont_env.simulate_batch_users_reward(batch_user_ids= user_ids, batch_recos=recos)
            # Update policy based on rewards
            policies[j].update_policy(user_ids, recos, rewards, probb, args.l_init)
            overall_rewards[j,i] = rewards.sum()
        # Print info
        if i == 0 or (i+1) % print_every == 0 or i+1 == n_rounds:
            logger.info("Round: %d/%d. Elapsed time: %f sec." % (i+1, n_rounds, time.time() - start_time))
            logger.info("Cumulative regrets: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(np.sum(overall_optimal_reward - overall_rewards[j]))) for j in range(n_policies)]))


    # Save results

    # logger.info("Saving cumulative regrets in %s" % args.output_path)
    # cumulative_regrets = {policies_name[j] : list(np.cumsum(overall_optimal_reward - overall_rewards[j])) for j in range(n_policies)}
    # with open(args.output_path, 'w') as fp:
    #     json.dump(cumulative_regrets, fp)

    logger.info("Saving cumulative regrets in %s" % args.output_path)

    # Check if the output file exists and is not empty
    if os.path.isfile(args.output_path) and os.path.getsize(args.output_path) > 0:
        # Load the existing data from the file
        with open(args.output_path, 'r') as fp:
            existing_data = json.load(fp)
            # Add the new data to the existing dictionary
            cumulative_regrets = {"{}".format(policies_name[j]): list(np.cumsum(overall_optimal_reward - overall_rewards[j])) for j
                                  in range(n_policies) if j != "random"}
    else:
        # Create a new empty dictionary if the file doesn't exist or is empty
        existing_data = {}
        # Add the new data to the existing dictionary
        cumulative_regrets = {"{}".format(policies_name[j]): list(np.cumsum(overall_optimal_reward - overall_rewards[j])) for j
                              in range(n_policies)}
    existing_data.update(cumulative_regrets)

    # Write the updated dictionary to the output file
    with open(args.output_path, 'w') as fp:
        json.dump(existing_data, fp)

