import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_util
import gym

import pickle
import numpy as np


tf.disable_v2_behavior()


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    return data


def norm(x):
    return (x - x.mean(0)) / x.std(0)


def build_model(X, y):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[X.shape[-1]]),
        layers.Dense(64, activation='relu'),
        layers.Dense(y.shape[-1])
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

# def to_tf(value):
#     return tf.convert_to_tensor(value, dtype=np.float32, dtype_hint=None, name=None)


def train(normed_train_data, train_labels):
    EPOCHS = 1000

    model = build_model(normed_train_data, train_labels)

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        # callbacks=[tfdocs.modeling.EpochDots()]
    )
    return history, model


def norm_obs(obs_bo, data):
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    policy_params = data[policy_type]

    # Build the policy. First, observation normalization.
    obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
    obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
    obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
    normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation
    return normedobs_bo

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('expert_norm_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    data = load_data(args.expert_data_file)
    X, y = data['observations'], data['actions']
    norm_data = load_data(args.expert_norm_file)
    normed_X = norm_obs(X[None, :], norm_data)

    with tf.Session():
        tf_util.initialize()

        # import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        history, model = train(normed_X.squeeze(), y.squeeze())
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                x_input = norm_obs(obs[None, :], norm_data)
                action = model.predict(x_input)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # expert_data = {'observations': np.array(observations),
        #                'actions': np.array(actions)}

        # with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        #     pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()