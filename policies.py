from collections import defaultdict
from math import log
from online_logistic_regression import OnlineLogisticRegression
from scipy.special import expit
from scipy.optimize import minimize
import numpy as np
from math import exp
from scipy.stats import bernoulli

# Abstract class defining the minimal functions that need
# to be implemented to create new bandit policy classes
class Policy:

    # Returns a list of size n_recos of playlist ids
    def recommend_to_users_batch(self, batch_users, n_recos=12):
        return

    # Updates policies parameters
    def update_policy(self, user_ids, recos, rewards):
        return


# A simple baseline that randomly recommends n_recos playlists to each user.
class RandomPolicy(Policy):
    def __init__(self, n_playlists, cascade_model=True):
        self.cascade_model = cascade_model
        self.n_playlists = n_playlists

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        n_users = len(batch_users)
        recos = np.zeros((n_users, n_recos), dtype=np.int64)
        r = np.arange(self.n_playlists)
        for i in range(n_users):
            np.random.shuffle(r)
            recos[i] = r[:n_recos]
        return recos

    def update_policy(self, user_ids, recos, rewards, probb, l_init=3):
        return

# Segment-based Thompson Sampling strategy, with Beta(alpha_zero,beta_zero) priors
class TSSegmentPolicy(Policy):
    def __init__(self, user_segment, user_segment_cascade, n_playlists, alpha_zero=1, beta_zero=99, model='cascade'):
        self.user_segment = user_segment
        self.user_segment_cascade = user_segment_cascade
        n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((n_segments, n_playlists))
        self.playlist_success = np.zeros((n_segments, n_playlists))
        self.alpha_zero = alpha_zero
        self.beta_zero = beta_zero
        self.t = 0
        self.cascade_model = model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_segment = np.take(self.user_segment, batch_users)
        user_segment_cascade = [self.user_segment_cascade[element] for element in user_segment]
        user_displays = np.take(self.playlist_display, user_segment, axis=0).astype(float)
        user_success = np.take(self.playlist_success, user_segment, axis=0)
        user_score = np.random.beta(self.alpha_zero + user_success, self.beta_zero + user_displays - user_success)
        user_choice = np.argsort(-user_score)[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice

    def update_policy(self, user_ids, recos, rewards, probb, l_init=3):
        for i, user_id in enumerate(user_ids):
            user_segment = self.user_segment[user_id]  # Indicate which segment the user belongs to

            if self.cascade_model == 'DCM':
                n_items = len(recos[i])
                self.attractiveness = probb[i]
                # examine = -(-0.091) * np.arange(n_items) + 1
                examine = np.exp(-0.99 * np.arange(n_items) )
                # examine = -0.6 * (np.arange(n_items)**2) + 1
                # examine[examine < 0] = 0
                Examination = np.full(n_items, -1)
                Click = np.zeros(n_items, dtype=bool)

                idx_init = np.arange(l_init - 1)
                Examination[idx_init] = 1
                # Click[idx_init] = bernoulli(self.attractiveness[idx_init]).rvs()
                Click[idx_init] = rewards[i][idx_init]

                idx_linit = l_init - 1
                Examination[idx_linit] = 1
                # Click[idx_linit] = bernoulli(self.attractiveness[idx_linit]).rvs()
                Click[idx_linit] = rewards[i][idx_linit]
                Examination[idx_linit + 1] = bernoulli(
                    1 - Click[idx_linit] + examine[idx_linit] * Click[idx_linit]).rvs()

                idx_else = np.arange(l_init, len(recos[i]))
                # Click[idx_else] = bernoulli(self.attractiveness[idx_else]).rvs()
                Click[idx_else] = rewards[i][idx_else]
                # distance = []
                # for pos in idx_else:
                #     if Click[pos] == 0 and any(Click[:pos]):
                #         distance.append((pos - np.where(Click[:pos] == 1)[0][0]))
                #     else:
                #         distance.append(pos)
                # distance = [(x - min(distance)) / (max(distance) - min(distance)) for x in distance]
                Examination[idx_else] = bernoulli(
                    1 - Click[idx_else - 1] + examine[idx_else - 1] * Click[idx_else - 1]).rvs()

            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for idx in range(len(recos[i])):
                p = recos[i][idx]
                r = rewards[i][idx]
                nb_display += 1
                self.playlist_success[user_segment][p] += r
                self.playlist_display[user_segment][p] += 1

                if self.cascade_model == 'cascade' and ((total_stream == 0 and nb_display == l_init) or (r == 1)):
                    break
                elif self.cascade_model == 'DCM' and (
                        (total_stream == 0 and nb_display == l_init) or not Examination[idx+1:].any()):
                    break
        return