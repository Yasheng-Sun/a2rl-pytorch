# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:26:11 2019

@author: yasheng sun
@email: yashengsun@sjtu.edu.cn
"""
import torch
import torch.nn as nn
import skimage.io as io
import skimage.transform as transform
import glob
import os
import numpy as np
import tensorflow as tf
import argparse
import vfn_network as vfn_net
from actions import command2action, generate_bbox, crop_input
import time
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from path import Path
import datetime

def build_mlp(input_size, n_layers, hidden_size, activation=nn.ReLU):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(input_size, hidden_size), activation()]
        input_size = hidden_size
    
    return nn.Sequential(*layers).apply(weights_init)

def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)

def pathlength(path):
    return len(path['reward'])

class AgentNet(nn.Module):
    def __init__(self, neural_network_args):
        super(AgentNet, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.hidden_size = neural_network_args['size']
        self.n_layers = neural_network_args['actor_n_layers']

        self.define_model_components()

    def define_model_components(self):
        self.mlp = build_mlp(self.ob_dim, self.n_layers, self.hidden_size)
        self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, 1)
        self.act = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.ac_dim))
        self.critic = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))

    def forward(self, ts_ob_no, state_tuple=None):
        # import pdb; pdb.set_trace();
        batch_size = ts_ob_no.size(0)
        mlp_out = self.mlp(ts_ob_no)
        if state_tuple == None:
            state_tuple = (torch.zeros(1, batch_size, self.hidden_size).to(ts_ob_no), 
                           torch.zeros(1, batch_size, self.hidden_size).to(ts_ob_no))
        _, next_state_tuple = self.encoder(mlp_out, state_tuple)
        ts_logits_na = self.act(next_state_tuple[0][0])
        ts_val = self.critic(next_state_tuple[0][0])
        return ts_logits_na, ts_val, next_state_tuple

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
    
    def forward(self, x):
        res = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        res = -1.0 * res.sum()
        return res

class Environment(object):
    def __init__(self, args):
        self.image_paths = glob.glob(os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), args.image_dir, '*.jpg'))
        self.ranking_loss = args.ranking_loss
        self.global_dtype_np = np.float32
        self.img_h, self.img_w = 227, 227
        
        embedding_dim = args.embedding_dim
        snapshot = args.snapshot
        net_data = np.load(args.initial_parameters, encoding='latin1').item()
        SPP = args.spp
        pooling = args.pooling
        global_dtype = tf.float32
        batch_size = 1
        self.image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size,self.img_h,self.img_w,3])
        var_dict = vfn_net.get_variable_dict(net_data)
        
        with tf.variable_scope("ranker") as scope:
            self.feature_vec = vfn_net.build_alexconvnet(self.image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
            self.score_func = vfn_net.score(self.feature_vec)

        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session(config=tf.ConfigProto())
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, snapshot)

    def score_feature(self, img):
        img = img.astype(np.float32) / 255
        img_resize = transform.resize(img, (self.img_h, self.img_w)) - 0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        score, feature = self.sess.run([self.score_func, self.feature_vec], feed_dict={self.image_placeholder: img_resize})

        return score, feature

    def reset(self):
        self.origin_img_path = self.image_paths[np.random.choice(len(self.image_paths))]
        self.origin_img = io.imread(self.origin_img_path)[:, :, :3]
        self.origin_score, self.origin_feature = self.score_feature(self.origin_img)
        
        return self.origin_feature, self.origin_score

    def step(self, action, steps, last_score, ratios, dones):
        ratios, dones = command2action([action], ratios, dones)
        bbox = generate_bbox(np.expand_dims(self.origin_img, axis=0), ratios)
        # import pdb; pdb.set_trace();
        next_img = crop_input(np.expand_dims(self.origin_img, axis=0), bbox)
        cur_score, next_ob = self.score_feature(next_img[0])
        rew = 1 if cur_score>last_score else -1
        rew -= 0.001*(steps+1)
        rew -= 5 if (bbox[0][2] - bbox[0][0]) > 2*(bbox[0][3] - bbox[0][1]) \
                or (bbox[0][2] - bbox[0][0]) < 0.5*(bbox[0][3] - bbox[0][1]) else 0
        return next_ob, rew, cur_score, ratios, dones

class Agent(object):
    def __init__(self, neural_network_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.hidden_size = neural_network_args['size']
        self.critic_n_layers = neural_network_args['critic_n_layers']
        self.actor_learning_rate = neural_network_args['actor_learning_rate']
        self.critic_learning_rate = neural_network_args['critic_learning_rate']
        self.num_target_updates = neural_network_args['num_target_updates']
        self.num_grad_steps_per_target_update = neural_network_args['num_grad_steps_per_target_update']
        self.beta = neural_network_args['entropy_loss_weight']

        self.max_path_length = sample_trajectory_args['max_path_length']
        self.gamma = estimate_advantage_args['gamma']
        self.agent_net = AgentNet(neural_network_args)

        # self.actor_optimizer = optim.Adam(self.agent_net.parameters(), lr=self.actor_learning_rate)
        # self.critic_optimizer = optim.Adam(self.agent_net.parameters(), lr=self.critic_learning_rate)
        self.optimizer = optim.Adam(self.agent_net.parameters(), lr=self.actor_learning_rate)

    def sample_action(self, ob_no, state_tuple=None):
        # import pdb; pdb.set_trace();
        ts_ob_no = torch.from_numpy(ob_no).float()
        ts_logits_na, ts_val, state_tuple = self.agent_net(ts_ob_no, state_tuple)
        ts_probs = nn.functional.log_softmax(ts_logits_na, dim = -1).exp()
        ts_sampled_ac = torch.multinomial(ts_probs, num_samples = 1).view(-1)

        sampled_ac = ts_sampled_ac.detach().numpy()
        return sampled_ac, state_tuple, ts_logits_na, ts_val
    
    @torch.no_grad()
    def get_value(self, ob_no, state_tuple):
        ts_ob_no = torch.from_numpy(ob_no).float()
        _, ts_val, _ = self.agent_net(ts_ob_no, state_tuple)
        return ts_val.detach().numpy()
    
    def sample_trajectory(self, env, origin_ob, cur_ob, score, state_tuple):
        ob = np.concatenate([origin_ob, cur_ob], axis=-1)
        obs, acs, rewards, next_obs, terminals, ts_logits_nas, ts_vals = [], [], [], [], [], [], []
        steps = 0
        ratios = np.repeat([[0, 0, 20, 20]], 1, axis=0)
        dones = np.array([0])

        while True:
            obs.append(ob)
            ac, state_tuple, ts_logits_na, ts_val = self.sample_action(ob[None], state_tuple)
            
            ts_logits_nas.append(ts_logits_na)
            ts_vals.append(ts_val)
            
            ac = ac[0]
            acs.append(ac)
            next_ob, rew, score, ratios, dones = env.step(ac, steps, score, ratios, dones)
            ob = np.concatenate([origin_ob, next_ob], axis=-1)
            next_obs.append(ob)
            rewards.append(rew)
            steps += 1

            if dones[0]:
                terminals.append(1)
                break
            else:
                terminals.append(0)
            
            if steps >= self.max_path_length:
                break

        terminal_val = 0 if dones[0] else self.get_value(ob[None], state_tuple)

        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32),
                "value": torch.stack(ts_vals).detach().numpy(),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32),
                "ts_logits_nas": torch.stack(ts_logits_nas),
                "ts_vals": torch.stack(ts_vals)}
        
        return path, state_tuple, dones[0], terminal_val, next_ob

    def get_log_prob(self, ts_logits_na, ts_ac_n):
        ts_logprob_n = torch.distributions.Categorical(logits=ts_logits_na).log_prob(ts_ac_n)
        return ts_logprob_n

    def update_parameters(self, path, adv_n, q_n):
        entropy_loss = HLoss()
        [ac_n, ts_logits_nas, ts_vals] = map(lambda x: path[x], ["action", "ts_logits_nas", "ts_vals"])

        ts_adv_n, ts_ac_n, ts_qn = map(lambda x: torch.from_numpy(x), [adv_n, ac_n, q_n])
        ts_logprob_n = self.get_log_prob(ts_logits_nas, ts_ac_n)
        
        actor_loss = - (ts_logprob_n*ts_adv_n).mean() + self.beta*entropy_loss(ts_logits_nas) # TODO: add entropy loss
        critic_loss = torch.nn.functional.mse_loss(ts_vals, ts_qn)
        
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward(retain_graph=True)
        self.optimizer.step()
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        metrics = {}
        metrics['actor_loss'] = actor_loss.item()
        metrics['critic_loss'] = critic_loss.item()
        metrics['tot_loss'] = loss.item()
        return metrics

    def estimate_advantage(self, path, terminal_val):
        path_len = pathlength(path)
        [re_n, val_n] = map(lambda x: path[x], ["reward", "value"])
        [q_n, adv_n] = map(lambda x: np.zeros_like(x), [re_n, re_n])
        for i in range(path_len-1, -1, -1):
            if(i==path_len-1):
                q_n[i] = self.gamma*terminal_val + re_n[i]
            else:
                q_n[i] = self.gamma*q_n[i+1] + re_n[i]
            adv_n[i] = q_n[i] - val_n[i]

        return q_n, adv_n

def train_AC(
        env,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        actor_learning_rate,
        critic_learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        logdir, 
        normalize_advantages,
        seed,
        actor_n_layers,
        critic_n_layers,
        ob_dim,
        ac_dim,
        size,
        entropy_loss_weight):
    neural_network_args = {
        'actor_n_layers': actor_n_layers,
        'critic_n_layers': critic_n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'size': size,
        'actor_learning_rate': actor_learning_rate,
        'critic_learning_rate': critic_learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
        'entropy_loss_weight': entropy_loss_weight,
        }

    sample_trajectory_args = {
        'max_path_length': max_path_length,
    }

    estimate_advantage_args = {
        'gamma': gamma,
    }

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = Agent(neural_network_args, sample_trajectory_args, estimate_advantage_args)

    for iter in range(n_iter):
        print("--------------iter {}--------------".format(iter))
        origin_ob, score = env.reset()
        state_tuple = None
        cur_ob = origin_ob
        t = 0
        while True:
            path, state_tuple, done, terminal_val, cur_ob = agent.sample_trajectory(env, origin_ob, cur_ob, score, state_tuple)
            t += pathlength(path)
            [q_n, adv_n] = agent.estimate_advantage(path, terminal_val)
            metrics = agent.update_parameters(path, adv_n, q_n)

            if done or t > min_timesteps_per_batch:
                break
        metrics['steps'] = t
        print("metrics:")
        val = ' '.join(['{}: {:.4f}'.format(key, metrics[key]) for key in metrics.keys()])

        print(val)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=50) ## Correspond to T_{max}
    parser.add_argument('--ep_len', '-ep', type=float, default=10) ## Correspond to t_{max}
    parser.add_argument('--actor_learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--critic_learning_rate', '-clr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--actor_n_layers', '-l', type=int, default=2)
    parser.add_argument('--critic_n_layers', '-cl', type=int)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument("--embedding_dim", help="Embedding dimension before mapping to one-dimensional score", type=int, default = 1000)
    parser.add_argument("--initial_parameters", help="Path to initial parameter file", type=str, default="alexnet.npy")
    parser.add_argument("--ranking_loss", help="Type of ranking loss", type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument("--snapshot", help="Name of the checkpoint files", type=str, default='./snapshots/model-spp-max')
    parser.add_argument("--spp", help="Whether to use spatial pyramid pooling in the last layer or not", type=bool, default=True)
    parser.add_argument("--pooling", help="Which pooling function to use", type=str, choices=['max', 'avg'], default='max')
    parser.add_argument("--image-dir", type=str, default='test_images')
    parser.add_argument("--ac_dim", type=int, default=14)
    parser.add_argument("--ob_dim", type=int, default=2000)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.05)
    parser.add_argument("--save_path", type=str, default='checkpoint')

    args = parser.parse_args()
    # timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    # training_writer = SummaryWriter(Path(args.save_path)/timestamp)

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = 'ac_' + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = Environment(args)
    
    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        train_AC(
            env=env,
            n_iter=args.n_iter,
            gamma=args.discount,
            min_timesteps_per_batch=args.batch_size,
            max_path_length=max_path_length,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            num_target_updates=args.num_target_updates,
            num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
            logdir=os.path.join(logdir,'%d'%seed),
            normalize_advantages=not(args.dont_normalize_advantages),
            seed=seed,
            actor_n_layers=args.actor_n_layers,
            critic_n_layers=args.critic_n_layers,
            ob_dim=args.ob_dim,
            ac_dim=args.ac_dim,
            size=args.size,
            entropy_loss_weight=args.entropy_loss_weight
        )

if __name__ == "__main__":
    main()