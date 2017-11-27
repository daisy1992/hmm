import numpy as np
import pandas as pd


def split_cluster(data):
    data.dropna(inplace=True)

    if data.shape[0] == 0:
        raise AssertionError()

    data['ENTRY_TIME'] = pd.to_datetime(data['ENTRY_TIME'])
    data['WEEK'] = data['ENTRY_TIME'].dt.week

    gb = data.groupby(['WEEK', 'CARD_NUMBER'])
    b = [gb.get_group(x) for x in gb.groups]

    return (
        np.array([d[['END', 'DURATION', 'DEST_X', 'DEST_Y']].values for d in b]),
        np.array([d['WEEKDAY'].values for d in b])
    )


def split(df, by):
    """
    Splits data according to a given parameter.
    :param df: Pandas DataFrame to split.
    :param by: Column on which to perform the split.
    :return: An array of arrays. Data is split on the first axis.
    """
    gb = df.groupby([by, 'CARD_NUMBER'])
    b = [gb.get_group(x) for x in gb.groups]

    l = []

    for d in b:
        batch = d[['END', 'DURATION', 'DEST_X', 'DEST_Y']].values

        l += [(batch, d['WEEKDAY'].values)]

        # if batch.shape[0] < 20:
        #     l += [(batch, d['WEEKDAY'].values)]

    # return (
    #     np.array([d[['END', 'DURATION', 'DEST_X', 'DEST_Y']].values for d in b]),
    #     np.array([d['WEEKDAY'].values for d in b])
    # )

    return (
        np.array([element[0] for element in l]),
        np.array([element[1] for element in l])
    )


def split_by_user(df, key=None):
    by = ['CARD_NUMBER']
    drop = ['CARD_NUMBER']

    if key:
        by += [key]

    grouped = df.reset_index().groupby(by=by)

    drop += ['WEEK']

    return np.array([group.drop(drop, axis=1).values for _, group in grouped])


def split_data(data):
    """
    Splits data according to a parameter hardcoded in the method. Calls the split() method.
    :param data: DataFrame to split.
    :return: An array of arrays. Data is split on the first axis.
    """
    data.dropna(inplace=True)

    if data.shape[0] == 0:
        raise AssertionError()

    return split(data, 'WEEK')


class CHmm:
    def __init__(self, n_components=4, n_clusters=8, n_iter=100, threshold=.01, full_cov=True, seed=None, verbose=True):
        # We seed the random number generator
        np.random.seed(seed)

        # Model parameters
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.threshold = threshold
        self.full_cov = full_cov
        self.verbose = verbose

        # Transition Matrix
        self.transition_matrix = np.random.rand(n_components, n_components)
        self.transition_matrix /= self.transition_matrix.sum(axis=1).reshape((-1, 1))

        # Vector of initial probabilities - P(x_0 = i)
        self.initial_probabilities = np.random.rand(n_components)
        self.initial_probabilities /= self.initial_probabilities.sum()

        # Emission Matrix
        self.emission_matrix = np.random.rand(n_components, n_clusters)
        self.emission_matrix /= self.emission_matrix.sum(axis=1).reshape((-1, 1))

        # Initialization of the model's parameters
        # Ante/post : before or after the cut-off time.
        self.mu_ante = None  # mu[k][d]
        self.mu_post = None  # mu[k][d]
        self.sigma_ante = None  # sigma[k][d, d]
        self.sigma_post = None  # sigma[k][d, d]

        self.F = None  # F[m][t, k]
        self.B = None  # B[m][t, i]
        self.Alpha = None  # Alpha[m][t, i]
        self.Beta = None  # Beta[m][t, i]
        self.Transition = None  # Transition[m][t, i, j]    (t=0..T-2, #t = T-1)
        self.State = None  # State[m][t, i]
        self.Cluster = None  # C[m][t, i, k]

        self.normalizing_factor = None

        # Dimension of the observations
        self.D = 0
        # Number of split
        self.M = 0

    def gaussian_likelihood(self, obs_batches):
        """
        Writes the likelihood of every observation. Corresponds to f(o | mu_k, Sigma_k).
        Returns nothing. The array F[m][t, k] is updated.
        :param obs_batches: Batches of observations - a M-dimensional array of batches of observations. 
        :return: An updated self.F
        """

        def mth_f(m, switch):

            if switch == 'ante':
                mu = self.mu_ante.copy()
                sigma = self.sigma_ante.copy()
                obs = obs_batches[m][:, :-2]
            elif switch == 'post':
                mu = self.mu_post.copy()
                sigma = self.sigma_post.copy()
                obs = obs_batches[m][:, -2:]
            else:
                raise IndexError(str(switch) + ' unknown.')

            def kth_cluster(k):
                diff = mu[k] - obs

                c = np.matmul(
                    np.expand_dims(diff, axis=1),
                    np.matmul(
                        np.linalg.inv(sigma[k]),
                        np.expand_dims(diff, axis=1).swapaxes(1, 2))
                ).reshape(-1)

                c = np.exp(-.5 * c) * 1 / (2 * np.pi) ** (.5 * self.D)

                return c

            f = np.array([kth_cluster(k) for k in range(self.n_clusters)]).swapaxes(0, 1)
            f /= np.expand_dims(np.linalg.det(sigma), axis=0) ** .5

            f = np.nan_to_num(f)
            return f

        self.F = np.array([mth_f(m, 'ante') * mth_f(m, 'post') for m in range(self.M)])
        return self.F

    def b(self):
        """
        Updates B[m][t, i], which corresponds to the output probability that the t-th observation from batch m
        is observed in state i.
        :return: An updated self.B
        """
        self.B = np.array([
            (np.expand_dims(self.emission_matrix, 0) * np.expand_dims(self.F[m], 1)).sum(axis=2) for m in range(self.M)
        ])

        self.B = np.nan_to_num(self.B)

        return self.B

    def forward_pass(self, obs_batches):
        """
        Updates Alpha[m][t, i], the 'alphas' of the forward pass. 
        Alpha[m][t, j] = B[m][t, j] \sum_i Alpha[m][t-1, i] * A[i, j]
        :param obs_batches: M-dimensional array, batches of observations.
        :return: An updated self.Alpha
        """

        normalizing_factor = []

        def mth_alpha(m, normalizer):
            alphas = np.zeros((obs_batches[m].shape[0], self.n_components))

            normalizers = np.zeros(obs_batches[m].shape[0])

            alphas[0, :] = self.initial_probabilities * self.B[m][0]
            normalizers[0] = alphas[0, :].sum()

            # if normalizers[0] == 0:
            #     alphas[0, :] += .25
            #     normalizers[0] = 1

            # print(str(m) + ': ' + str(alphas[0, :]))

            alphas[0, :] /= normalizers[0]

            for t in range(1, obs_batches[m].shape[0]):
                alphas[t, :] = self.B[m][t] * np.sum(alphas[t - 1, :] * self.transition_matrix, axis=1)
                normalizers[t] = alphas[t, :].sum()

                # if normalizers[t] == 0:
                #     alphas[t, :] += .25
                #     normalizers[t] = 1

                alphas[t, :] /= normalizers[t]

            normalizer += [normalizers]

            alphas = np.nan_to_num(alphas)
            return alphas

        self.Alpha = np.array([mth_alpha(m, normalizing_factor) for m in range(self.M)])
        self.normalizing_factor = np.array(normalizing_factor)

        return self.Alpha

    def backward_pass(self, obs_batches):
        """
        Updates Beta[m][t, i], the 'betas' of the forward pass. 
        :param obs_batches: M-dimensional array, batches of observations.
        :return: An updated self.Beta
        """

        def mth_beta(m):
            betas = np.zeros((obs_batches[m].shape[0], self.n_components))

            betas[-1, :] = 1

            for t in range(obs_batches[m].shape[0] - 2, -1, -1):
                betas[t, :] = np.sum(self.B[m][t + 1] * betas[t + 1, :] * self.transition_matrix, axis=1) \
                              / self.normalizing_factor[m][t+1]

            betas = np.nan_to_num(betas)
            return betas

        self.Beta = np.array([mth_beta(m) for m in range(self.M)])
        return self.Beta

    def transition_probability(self):
        """
        Computes and updates Transition[m][t, i, j], which corresponds to P(x_t = i, x_{t+1} = j | o)
        :return: An updated self.Transition
        """

        def mth_t(m):
            t = np.expand_dims(self.Alpha[m][:-1, :], axis=2) * np.expand_dims(self.transition_matrix, axis=0) * \
                np.expand_dims(self.Beta[m] * self.B[m], axis=1)[1:, :, :]

            return np.nan_to_num(t / np.expand_dims(np.expand_dims(t.sum(axis=(1, 2)), 1), 2))

        self.Transition = np.array([mth_t(m) for m in range(self.M)])
        return self.Transition

    def state_probability(self):
        """
        Computes and updates State[m][t, i], which corresponds to P(x_t = i | o)
        :return: An updated self.State
        """

        def mth_s(m):
            s = self.Alpha[m] * self.Beta[m]
            return np.nan_to_num(s / np.expand_dims(s.sum(axis=1), 1))

        self.State = np.array([mth_s(m) for m in range(self.M)])
        return self.State

    def cluster_probability(self):
        """
        Computes and updates Cluster[m][t, i, k], which corresponds to P(x_t = i, m_t = k | o)
        :return: An updated self.Cluster
        """

        def mth_c(m):
            # print(np.expand_dims(self.State[m] / self.B[m], axis=2))
            # print(self.emission_matrix)
            # print(np.expand_dims(self.F[m], axis=1))

            c = np.expand_dims(self.State[m] / self.B[m], axis=2) * \
                np.expand_dims(self.emission_matrix, axis=0) * \
                np.expand_dims(self.F[m], axis=1)

            c = np.nan_to_num(c)

            return c

        self.Cluster = np.array([mth_c(m) for m in range(self.M)])
        return self.Cluster

    def log_likelihood(self, obs_batches):
        """
        Writes the likelihood of every observation. Corresponds to f(o | mu_k, Sigma_k).
        Returns nothing. The array F[m][t, k] is updated.
        :param obs_batches: Batches of observations - a M-dimensional array of batches of observations.
        :return: An updated self.F
        """

        def mth_f(m, switch):

            if switch == 'ante':
                mu = self.mu_ante
                sigma = self.sigma_ante
                obs = obs_batches[m][:, :-2]
            elif switch == 'post':
                mu = self.mu_post
                sigma = self.sigma_post
                obs = obs_batches[m][:, -2:]
            else:
                raise IndexError(str(switch) + ' unknown.')

            def kth_cluster(k):
                diff = mu[k] - obs

                c = np.matmul(
                    np.expand_dims(diff, axis=1),
                    np.matmul(
                        np.linalg.inv(sigma[k]),
                        np.expand_dims(diff, axis=1).swapaxes(1, 2))
                ).reshape(-1)

                c = -.5 * c + np.log(1 / (2 * np.pi) ** (.5 * self.D))

                return c

            f = np.array([kth_cluster(k) for k in range(self.n_clusters)]).swapaxes(0, 1)
            f -= np.expand_dims(np.log(np.linalg.det(sigma)), axis=0) / 2

            f = np.nan_to_num(f)

            return f

        result = np.array([(mth_f(m, 'ante') * mth_f(m, 'post')).sum() for m in range(self.M)])
        # result = np.log(result).sum()
        # result = np.nan_to_num(result)

        return result.sum()

    def expectation(self, obs_batches):
        """
        Performs the expectation step of the Baum-Welch/EM algorithm.
        Updates F, B, Alpha, Beta, Transition, State and Cluster.
        :param obs_batches: M-dimensional array, batches of observations.
        :return: None.
        """
        self.gaussian_likelihood(obs_batches)
        self.b()
        self.forward_pass(obs_batches)
        self.backward_pass(obs_batches)
        self.transition_probability()
        self.state_probability()
        self.cluster_probability()

    def maximization(self, obs_batches, full_cov=True):
        """
        Performs the maximization step.
        :param obs_batches: M-dimensional array, batches of observations.
        :param full_cov: boolean, whether the covariance matrices are full or diagonal.
        :return: None.
        """

        # State[m][t, i]
        # initial_probabilities = np.array([state[0] for state in self.State]).sum(axis=0)  # / self.M
        # print(initial_probabilities)
        initial_probabilities = np.ones(self.n_components)
        initial_probabilities /= initial_probabilities.sum()

        # Transition[m][t, i, j]
        transition_matrix = np.array([transition.sum(axis=0) for transition in self.Transition]).sum(axis=0)
        transition_matrix /= np.expand_dims(
            np.array([state[:-1, :].sum(axis=0) for state in self.State]).sum(axis=0),
            1
        )

        emission_matrix = np.array([cluster.sum(axis=0) for cluster in self.Cluster]).sum(axis=0)
        emission_matrix /= np.expand_dims(
            np.array([state.sum(axis=0) for state in self.State]).sum(axis=0),
            1
        )
        emission_matrix /= np.expand_dims(emission_matrix.sum(axis=1), 1)

        denominator = np.expand_dims(
            np.array([cluster.sum(axis=(0, 1)) for cluster in self.Cluster]).sum(axis=0),
            1
        )

        mu_ante = np.nan_to_num(np.array([
            (np.expand_dims(self.Cluster[m].sum(axis=1), axis=2) * np.expand_dims(obs_batches[m][:, :-2], 1)).sum(0)
            for m in range(self.M)
        ]).sum(axis=0) / denominator)

        mu_post = np.nan_to_num(np.array([
            (np.expand_dims(self.Cluster[m].sum(axis=1), axis=2) * np.expand_dims(obs_batches[m][:, -2:], 1)).sum(0)
            for m in range(self.M)
        ]).sum(axis=0) / denominator)

        def kth_covariance(k, switch):

            def mth_var(m):  # var[m][t, d1, d2]

                if switch == 'ante':
                    diff = obs_batches[m][:, :-2] - mu_ante[k]  # diff[t, d]
                elif switch == 'post':
                    diff = obs_batches[m][:, -2:] - mu_post[k]  # diff[t, d]
                else:
                    raise IndexError(str(switch) + ' unknown.')

                if full_cov:
                    return np.array([d.reshape(-1, 1).dot(d.reshape(-1, 1).T) for d in diff])
                else:
                    return np.array([np.diag(d ** 2) for d in diff])

            covariance = np.nan_to_num(np.array([
                (mth_var(m) * self.Cluster[m][:, :, k].sum(axis=1).reshape(-1, 1, 1)).sum(axis=0) for m in range(self.M)
            ]).sum(axis=0))

            return covariance

        sigma_ante = np.array(
            [kth_covariance(k, 'ante') for k in range(self.n_clusters)]
        ) / np.array(
            [cluster.sum(axis=(0, 1)) for cluster in self.Cluster]
        ).sum(axis=0).reshape(-1, 1, 1)
        sigma_ante += np.expand_dims(np.diag([.1] * (obs_batches[0].shape[1] - 2)), 0)

        sigma_post = np.array(
            [kth_covariance(k, 'post') for k in range(self.n_clusters)]
        ) / np.array(
            [cluster.sum(axis=(0, 1)) for cluster in self.Cluster]
        ).sum(axis=0).reshape(-1, 1, 1)
        sigma_post += np.expand_dims(np.diag([.1] * 2), 0)

        score = np.absolute(initial_probabilities - self.initial_probabilities).mean() / 7.
        score += np.absolute(transition_matrix - self.transition_matrix).mean() / 7.
        score += np.absolute(emission_matrix - self.emission_matrix).mean() / 7.

        score += np.absolute(mu_ante - self.mu_ante).mean() / 7.
        score += np.absolute(mu_post - self.mu_post).mean() / 7.
        score += np.absolute(sigma_ante - self.sigma_ante).mean() / 7.
        score += np.absolute(sigma_post - self.sigma_post).mean() / 7.

        self.initial_probabilities = np.nan_to_num(initial_probabilities)
        self.transition_matrix = np.nan_to_num(transition_matrix)
        self.emission_matrix = np.nan_to_num(emission_matrix)

        self.mu_ante = np.nan_to_num(mu_ante)
        self.mu_post = np.nan_to_num(mu_post)
        self.sigma_ante = np.nan_to_num(sigma_ante)
        self.sigma_post = np.nan_to_num(sigma_post)

        return score

    def initiate(self, obs_batches):
        self.M = obs_batches.shape[0]
        self.D = obs_batches[0].shape[1]

        self.mu_ante = np.zeros((self.n_clusters, self.D - 2))
        self.mu_post = np.zeros((self.n_clusters, 2))

        if self.full_cov:
            self.sigma_ante = np.array(
                [10 * np.ones((self.D-2, self.D-2)) + 1000 * np.identity(self.D-2)] * self.n_clusters)
            self.sigma_post = np.array(
                [10 * np.ones((2, 2)) + 100 * np.identity(2)] * self.n_clusters)
        else:
            self.sigma_ante = np.array([1000 * np.identity(self.D-2)] * self.n_clusters)
            self.sigma_post = np.array([100 * np.identity(2)] * self.n_clusters)

    def step(self, obs_batches):
        self.expectation(obs_batches)
        diff = self.maximization(obs_batches, full_cov=self.full_cov)
        return diff

    def fit(self, obs_batches):
        """
        Performs the fitting step, and learns the parameters from the data.
        :param obs_batches: M-dimensional array, batches of observations.
        :return: the number of iterations before convergence.
        """
        self.M = obs_batches.shape[0]
        self.D = obs_batches[0].shape[1]

        if self.mu_ante is None:
            self.mu_ante = np.random.normal(
                np.concatenate(obs_batches)[:2, :].mean(),
                np.concatenate(obs_batches)[:2, :].std(),
                (self.n_clusters, self.D-2)
            )
            self.mu_post = np.random.normal(
                np.concatenate(obs_batches)[2:, :].mean(),
                np.concatenate(obs_batches)[2:, :].std(),
                (self.n_clusters, self.D-2)
            )

        if self.full_cov:
            self.sigma_ante = np.array(
                [10 * np.ones((self.D-2, self.D-2)) + 1000 * np.identity(self.D-2)] * self.n_clusters)
            self.sigma_post = np.array(
                [10 * np.ones((2, 2)) + 100 * np.identity(2)] * self.n_clusters)
        else:
            self.sigma_ante = np.array([1000 * np.identity(self.D-2)] * self.n_clusters)
            self.sigma_post = np.array([100 * np.identity(2)] * self.n_clusters)

        for i in range(self.n_iter):
            self.expectation(obs_batches)
            diff = self.maximization(obs_batches, full_cov=self.full_cov)

            if self.verbose:
                print(str(i + 1) + '-th pass : ' + str(diff), end='\r')

            if diff < self.threshold:
                print()
                return i

        print()
        return 0

    def viterbi(self, observations):
        """
        Performs the decoding task. Given a learned HMM, computes the most likely sequence of states resulting on
        the supplied sequence of observations.
        :param observations: T*D array of observations.
        :return: The most likely sequence of states.
        """

        def ante_post(switch):

            if switch == 'ante':
                mu = self.mu_ante
                sigma = self.sigma_ante
                obs = observations[:, :-2]
            elif switch == 'post':
                mu = self.mu_post
                sigma = self.sigma_post
                obs = observations[:, -2:]
            else:
                raise IndexError(str(switch) + ' unknown.')

            def kth_cluster(k):
                diff = mu[k] - obs

                c = np.matmul(
                    np.expand_dims(diff, axis=1),
                    np.matmul(
                        np.linalg.inv(sigma[k]),
                        np.expand_dims(diff, axis=1).swapaxes(1, 2))
                ).reshape(-1)

                c = np.exp(-.5 * c) * 1 / (2 * np.pi) ** (.5 * self.D)

                return c

            res = np.array([kth_cluster(k) for k in range(self.n_clusters)]).swapaxes(0, 1)
            res /= np.expand_dims(np.linalg.det(sigma), axis=0) ** .5

            return res

        f = ante_post('ante') * ante_post('post')

        b = (np.expand_dims(self.emission_matrix, 0) * np.expand_dims(f, 1)).sum(axis=2)

        delta = np.zeros((observations.shape[0], self.n_components))
        psi = np.zeros((observations.shape[0], self.n_components))

        delta[0, :] = self.initial_probabilities * b[0, :]

        for t in range(1, observations.shape[0]):
            delta[t, :] = b[t] * (np.expand_dims(delta[t - 1, :], 1) * self.transition_matrix).max(axis=0)
            psi[t, :] = (np.expand_dims(delta[t - 1, :], 1) * self.transition_matrix).argmax(axis=0)

        x = np.zeros(observations.shape[0]).astype(int)

        x[-1] = delta[-1, :].argmax()

        for t in range(observations.shape[0] - 2, -1, -1):
            x[t] = psi[t + 1, x[t + 1]]

        return x

    def summary(self):
        print()
        print('-----SUMMARY-----')
        print('Initial Probabilities:')
        initial_probabilities = self.initial_probabilities.copy()
        initial_probabilities[initial_probabilities < .01] = 0
        print(initial_probabilities)
        print()
        print('Transition Matrix:')
        transition_matrix = self.transition_matrix.copy()
        transition_matrix[transition_matrix < .01] = 0
        print(transition_matrix)
        print()
        print('Emission Matrix:')
        emission_matrix = self.emission_matrix.copy()
        emission_matrix[emission_matrix < .01] = 0
        print(emission_matrix)
        print()
        print('Max on Emission Matrix:')
        emission_matrix = self.emission_matrix.copy()
        emission_matrix[emission_matrix < .01] = 0
        print(emission_matrix.max(axis=0))
        print()
        print('Means & Covariances of Gaussians:')
        mu_ante = self.mu_ante.copy()
        mu_ante[np.absolute(mu_ante) < .01] = 0
        mu_post = self.mu_post.copy()
        mu_post[np.absolute(mu_post) < .01] = 0

        sigma_ante = self.sigma_ante.copy()
        sigma_ante[sigma_ante < .01] = 0
        sigma_post = self.sigma_post.copy()
        sigma_post[sigma_post < .01] = 0
        for k in range(self.n_clusters):
            print()
            print(str(k + 1) + '-th gaussian:')
            print(mu_ante[k])
            print(sigma_ante[k])
            print(mu_post[k])
            print(sigma_post[k])

    def most_likely_state(self, observations):
        """
        Returns the most likely current state given past and current observations.
        max_i P(x_t = i | o_{1:t})
        :param observations: A sequence of past observations.
        :return: The most likely current state -- the one that emitted the last observation.
        """

        self.M = 1

        self.gaussian_likelihood(np.expand_dims(observations, axis=0))
        self.b()
        self.forward_pass(np.expand_dims(observations, axis=0))

        return np.argmax(self.Alpha[0][-1, :])

    def next_state_probability(self, observations):
        """
        Returns the most likely current state given past and current observations.
        max_i P(x_t = i | o_{1:t})
        :param observations: A sequence of past observations.
        :return: The most likely current state -- the one that emitted the last observation.
        """

        self.M = 1

        self.gaussian_likelihood(np.expand_dims(observations, axis=0))
        self.b()
        self.forward_pass(np.expand_dims(observations, axis=0))
        self.backward_pass(np.expand_dims(observations, axis=0))

        res = np.sum(self.transition_matrix * np.expand_dims(self.Alpha[0][-1, :] * self.Beta[0][-1, :], 0), axis=1)

        return res / res.sum()

    def predict(self, observations):
        """
        Makes a prediction about the next place.
        :param observations: A sequence of past observations.
        :return: A prediction about the next place, corresponding to the mean of the most likely cluster.
        """

        # next_state[n] gives the probability of being in the n-th state at time t+1
        next_state = self.next_state_probability(observations)

        # cluster_probabilities[k] gives the probability of being in the k-th cluster at time t+1
        cluster_probabilities = np.sum(
            np.expand_dims(next_state, 1) * self.emission_matrix,
            axis=0
        )
        cluster_probabilities /= cluster_probabilities.sum()

        return np.concatenate(
            (self.mu_ante[np.argmax(cluster_probabilities)], self.mu_post[np.argmax(cluster_probabilities)])
        )

    def s_predict(self, observations, possible_destinations_positions):
        """
        Makes a prediction about the next place, using information about the stations.
        :param possible_destinations_positions: stations accessible from the network
        :param observations: A sequence of past observations.
        :return: A prediction about the next place - the position of the most likely exit station.
        """

        # next_state[n] gives the probability of being in the n-th state at time t+1
        next_state = self.next_state_probability(observations)

        # cluster_probabilities[k] gives the probability of being in the k-th cluster at time t+1
        cluster_probabilities = np.sum(
            np.expand_dims(next_state, 1) * self.emission_matrix,
            axis=0
        )
        cluster_probabilities /= cluster_probabilities.sum()

        def kth_cluster(k):
            """
            Computes the density of the k-th cluster at every possible exit station.
            :param k: The considered cluster.
            :return: An p-dimensional array of likelihoods, p being the number of possible exit stations.
            """
            diff = self.mu_post[k] - possible_destinations_positions.values

            c = np.matmul(
                np.expand_dims(diff, axis=1),
                np.matmul(
                    np.linalg.inv(self.sigma_post[k]),
                    np.expand_dims(diff, axis=2)
                )
            ).reshape(-1)

            c = np.exp(-.5 * c) * 1 / (2 * np.pi) * 1 / (np.linalg.det(self.sigma_post[k]) ** .5)

            return c

        f = np.array([kth_cluster(k) for k in range(self.n_clusters)])

        f *= np.expand_dims(cluster_probabilities, 1)

        best = np.argmax(f.sum(axis=0))

        return np.array([
            0,
            0,
            possible_destinations_positions.values[best][0],
            possible_destinations_positions.values[best][1]
        ])

    def cluster_probabilities(self, observations, next_observation):
        """
        Computes the probability of each cluster given the sequence of past observations, and the new entry information.
        :param observations: n x d matrix, where n is the number of past observations and d their dimension
        :param next_observation: the new entry information
        :return: a K-dimensional array giving the probability of each cluster
        """

        self.M = 1

        self.gaussian_likelihood(np.expand_dims(observations, 0))
        self.b()
        self.forward_pass(np.expand_dims(observations, 0))

        def kth_cluster(k):
            """
            Computes the likelihood of the last observation for the k-th cluster.
            :param k: The cluster that is being tested.
            :return: The probability of each cluster.
            """

            diff = self.mu_ante[k] - next_observation[:2]

            c = diff.dot(np.linalg.inv(self.sigma_ante[k]).dot(diff))
            c = np.exp(-.5 * c) * 1 / (2 * np.pi) * 1 / (np.linalg.det(self.sigma_ante[k]) ** .5)

            d = self.transition_matrix * np.expand_dims(self.Alpha[0][-1], 1)
            d *= np.expand_dims(self.emission_matrix[:, k], 0)

            return c * d.sum()

        res = np.array([kth_cluster(k) for k in range(self.n_clusters)])
        return res / res.sum()

    def ss_predict(self, observations, possible_destinations_positions, next_observation):
        """
        Makes a prediction about the next place, using information about the stations.
        :param next_observation: information about the position...
        :param possible_destinations_positions: stations accessible from the network
        :param observations: A sequence of past observations.
        :return: A prediction about the next place.
        """

        # cluster_probabilities gives the probability of each cluster
        cluster_probabilities = self.cluster_probabilities(observations, next_observation)

        def kth_cluster(k):
            """
            Computes the density of the k-th cluster at every possible exit station.
            :param k: The considered cluster.
            :return: An p-dimensional array of likelihoods, p being the number of possible exit stations.
            """
            diff = self.mu_post[k] - possible_destinations_positions.values

            c = np.matmul(
                np.expand_dims(diff, axis=1),
                np.matmul(
                    np.linalg.inv(self.sigma_post[k]),
                    np.expand_dims(diff, axis=2)
                )
            ).reshape(-1)

            c = np.exp(-.5 * c) * 1 / (2 * np.pi) * 1 / (np.linalg.det(self.sigma_post[k]) ** .5)

            return c

        f = np.array([kth_cluster(k) for k in range(self.n_clusters)])

        f *= np.expand_dims(cluster_probabilities, 1)

        best = np.argmax(f.sum(axis=0))

        return np.array([
            next_observation[0],
            next_observation[1],
            possible_destinations_positions.values[best][0],
            possible_destinations_positions.values[best][1]
        ])

    def prediction(self, observations, next_observation):
        """
        Makes a prediction about the next place, using information about the stations.
        :param next_observation: information about the position...
        :param observations: A sequence of past observations.
        :return: A prediction about the next place.
        """

        # cluster_probabilities gives the probability of each cluster
        cluster_probabilities = self.cluster_probabilities(observations, next_observation)

        best = np.argmax(cluster_probabilities)

        return self.mu_post[best]

    def score_clusters(self, x, test_time=True):
        """
        Return the log-likelihood of the x parameter (in terms of clusters).
        :param x: N*D array, where N is the number of observations and D their dimension.
        :param test_time: Whether to test the time as well (clusters give a time and place).
        :return: the log-likelihood of the observations.
        """

        def kth_cluster(k):
            """
            Computes the density of the k-th cluster at every possible exit station.
            :param k: The considered cluster.
            :param test_time: Whether to test the time as well (clusters give a time and place).
            :return: An p-dimensional array of likelihoods, p being the number of possible exit stations.
            """

            if test_time:
                diff = self.mu[k] - x

                c = np.matmul(
                    np.expand_dims(diff, axis=1),
                    np.matmul(
                        np.linalg.inv(self.sigma[k]),
                        np.expand_dims(diff, axis=2)
                    )
                ).reshape(-1)

                c = np.exp(-.5 * c) * 1 / (2 * np.pi) * 1 / (np.linalg.det(self.sigma[k]) ** .5)

            else:
                diff = self.mu[k][2:] - x[:, 2:]

                c = np.matmul(
                    np.expand_dims(diff, axis=1),
                    np.matmul(
                        np.linalg.inv(self.sigma[k][2:, 2:]),
                        np.expand_dims(diff, axis=2)
                    )
                ).reshape(-1)

                c = np.exp(-.5 * c) * 1 / (2 * np.pi) * 1 / (np.linalg.det(self.sigma[k][2:, 2:]) ** .5)

            return c

        likelihood = np.array(
            [kth_cluster(k) for k in range(self.mu.shape[0])]
        ).sum(axis=0)

        return np.log(likelihood).mean()

    def evaluation(self, observations):
        """
        Performs the evaluation task: returns the log-likelihood of the sequence of observations.
        :param observations: Sequence of observations to evaluate.
        :return: Log-likelihood of the sequence.
        """

        self.gaussian_likelihood(np.expand_dims(observations, axis=0))
        self.b()

        time_horizon = observations.shape[0]

        # We use the notations introduced for the Viterbi algorithm
        delta = np.zeros((time_horizon, self.n_components))

        # Initialization
        delta[0, :] = self.initial_probabilities * self.B[0][0]

        # Recursion
        for t in range(1, time_horizon):
            delta[t, :] = np.max(
                np.expand_dims(delta[t-1], axis=0) * self.transition_matrix * np.expand_dims(self.B[0][t], axis=1),
                axis=1
            )

        return np.log(np.max(delta[-1]))

    def score(self, obs_batches):
        """
        Performs an evaluation of every batch of observation and returns the log-likelihood of the entire sequence.
        :param obs_batches: Sequence of observation (in batches).
        :return: Log-likelihood of the sequence.
        """

        return sum(
            [self.evaluation(obs_batches[i]) for i in range(obs_batches.shape[0])]
        )

    @staticmethod
    def reseed():
        np.random.seed()

    def test_user(self, data):

        distances = []
        for k in range(-20, 0):
            try:
                prediction = self.prediction(data.values[:k, :], data.values[k, :2])
                distances += [((prediction - data.values[k, 2:]) ** 2).sum() ** .5]
            except:
                distances += [1000]

        return np.array(distances)

