import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import colorama
from tqdm import tqdm 

from torch.distributions import Categorical
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from pytracking.evaluation.data import Sequence
from ltr.data.image_loader import opencv_loader as image_loader
from ltr.data.sampler import RandomSequenceSampler

from .models import ADNet
from .utils import gen_samples, overlap_ratio, extract_region, RegionExtractor, SampleGenerator, plot_image


def identity_func(x): return x


def clamp_bbox(bbox, img_size):
    bbox[2:] = np.clip(bbox[2:], 10, img_size - 10)
    bbox[:2] = np.clip(bbox[:2], 0, img_size - bbox[2:] - 1)
    return bbox


def perform_action(bbox, action_idx, params):
    # Perform action on bounding box(es).
    # - bbox: 1d array of     [x,y,w,h] or
    #         2d array of N x [x,y,w,h]
    # - action_idx: scalar or array of size N (action for each bbox)
    is_1d = bbox.ndim == 1
    if is_1d:
        bbox = bbox[np.newaxis, :]

    opts = params.opts['action_move']
    x, y, w, h = [bbox[:, i] for i in range(4)]

    deltas = np.array([ opts['x'] * w, opts['y'] * h, opts['w'] * w, opts['h'] * h ]).T
    deltas = np.maximum(deltas, 1)
    
    aspect_ratio = w / h
    iff = w > h
    deltas[iff, 3] = deltas[iff, 2] / aspect_ratio[iff]
    deltas[~iff, 2] = deltas[~iff, 3] * aspect_ratio[~iff]
    # if w > h:
    #     deltas[3] = deltas[2] / aspect_ratio
    # else:
    #     deltas[2] = deltas[3] * aspect_ratio
    
    action_delta = opts['deltas'][action_idx, :] * deltas  # Element-wise product
    bbox_next = bbox.copy()
    bbox_next[:, :2] += 0.5 * bbox_next[:, 2:]  # Center
    bbox_next += action_delta
    bbox_next[:, :2] -= 0.5 * bbox_next[:, 2:]  # Un-center

    # NOTE: Original code performs clamping here (necessary? aspect ratio changes ...)

    if is_1d:
        return bbox_next.squeeze()
    return bbox_next


def generate_action_labels(bbox, samples, params):
    # Return the indexes of the best actions when moving samples towards bbox.

    # Calculate overlap between bbox and samples for all actions.
    overlaps = np.zeros((samples.shape[0], params.num_actions))
    for a in range(params.num_actions):
        overlaps[:, a] = overlap_ratio(bbox, perform_action(samples, a, params))
    
    # Check translation actions only
    max_values = np.max(overlaps[:, :-2], axis=1)
    max_actions = np.argmax(overlaps[:, :-2], axis=1)

    # Stop action if close enough
    stop_actions = overlaps[:, params.opts['stop_action']] > params.opts['stopIou']
    max_actions[stop_actions] = params.opts['stop_action']

    # Allow scaling actions if stop action is best
    stop_actions = max_values == overlaps[:, params.opts['stop_action']]
    max_actions[stop_actions] = np.argmax(overlaps[stop_actions], axis=1)

    return max_actions


# Outputs batches of positive/negative training samples for SL training.
# TODO precompute labels and use multiple workers if serious about training 
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequence, params):
        self.params = params
        self.seq = sequence
        self.frames = np.array(self.seq.frames)  # Image paths

        # Batch size = batch frames * (batch pos + batch neg)

        self.index = torch.randperm(len(self.seq.frames))  # Shuffle frames
        self.pointer = 0

        self.crop_size = params.img_size

        image = image_loader(self.seq.frames[0])
        img_size = np.array(image.shape[1::-1])
        self.pos_generator = SampleGenerator('gaussian', img_size,
                params.trans_pos, params.scale_pos)
        self.neg_generator = SampleGenerator('gaussian', img_size,
                params.trans_neg, params.scale_neg)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        idx = self.index[index]
        image = image_loader(self.frames[idx])
        bbox = self.seq.ground_truth_rect[idx]
        pos_samples = self.pos_generator(bbox, self.params.n_pos_train, overlap_range=self.params.overlap_pos_train)
        neg_samples = self.neg_generator(bbox, self.params.n_neg_train, overlap_range=self.params.overlap_neg_train)
        pos_patches = RegionExtractor(image, pos_samples, self.params)()
        neg_patches = RegionExtractor(image, neg_samples, self.params)()
        pos_action_labels = torch.from_numpy(generate_action_labels(bbox, pos_samples, self.params))
        pos_score_labels = torch.tensor(1).expand(pos_patches.size(0))
        neg_score_labels = torch.tensor(0).expand(neg_patches.size(0))
        return (pos_patches, pos_action_labels, pos_score_labels), (neg_patches, neg_score_labels)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.frames):
            self.pointer = 0
            raise StopIteration

        next_pointer = min(self.pointer + self.params.batch_frames, len(self.frames))
        idx = self.index[self.pointer:next_pointer]
        self.pointer = next_pointer
        
        data = [self[i] for i in idx]
        return SequenceDataset._collate(data)

    @staticmethod
    def _collate(data):
        pos_patches = torch.cat([d[0][0] for d in data], dim=0)
        pos_action_labels = torch.cat([d[0][1] for d in data], dim=0)
        pos_score_labels = torch.cat([d[0][2] for d in data], dim=0)
        neg_patches = torch.cat([d[1][0] for d in data], dim=0)
        neg_score_labels = torch.cat([d[1][1] for d in data], dim=0)
        return (pos_patches, pos_action_labels, pos_score_labels), (neg_patches, neg_score_labels)


def classification_accuracy(model, db, device, params):
    if len(db) == 0: return 0, 0

    model.eval()

    num_correct_actions = 0
    num_correct_scores = 0
    num_actions = 0
    num_scores = 0
    with torch.no_grad():
        for sequence in db:
            for pos_data, neg_data in SequenceDataset(sequence, params):
                pos_patches, pos_actions, pos_scores = pos_data
                neg_patches, neg_scores = neg_data
                
                # plot_image(pos_patches[0].permute(1, 2, 0).numpy().astype(np.uint8) + 128)
                # plot_image(neg_patches[0].permute(1, 2, 0).numpy().astype(np.uint8) + 128)
                # print(pos_actions[0])

                pos_patches = pos_patches.to(device) 
                pos_actions = pos_actions.to(device)
                pos_scores = pos_scores.to(device)

                neg_patches = neg_patches.to(device)
                neg_scores = neg_scores.to(device)

                action_history_oh_zero = torch.tensor(0.0).expand(pos_patches.size(0), model.action_history_size).to(device)
                out_actions, out_scores = model(pos_patches, action_history_oh_zero)
                num_correct_actions += (out_actions.argmax(dim=1) == pos_actions).sum().item()
                num_actions += len(pos_actions)
                num_correct_scores += (out_scores.argmax(dim=1) == pos_scores).sum().item()
                num_scores += len(pos_scores)

                action_history_oh_zero = torch.tensor(0.0).expand(neg_patches.size(0), model.action_history_size).to(device)
                _, out_scores = model(neg_patches, action_history_oh_zero)
                num_correct_scores += (out_scores.argmax(dim=1) == neg_scores).sum().item()
                num_scores += len(neg_scores)

    return num_correct_actions / num_actions, num_correct_scores / num_scores


def latest_checkpoint(path):
    checkpoints = list(path.glob("checkpoint_*.pth"))
    if len(checkpoints) > 0:
        epoch = max(map(int, (p.stem.rsplit('_')[1] for p in checkpoints)))
        return path / f"checkpoint_{epoch}.pth"
    else:
        return None


class TrackerTrainer:
    def __init__(self, params, model_path=None, epoch_checkpoint=1, epochs=10, evaluate_performance_=True, experiment_name=""):
        self.params = params 
        self.epoch_checkpoint = epoch_checkpoint
        self.n_epochs = epochs
        self.evaluate_performance_ = evaluate_performance_
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_path = params.checkpoints_path / experiment_name
        self.stats = dict(total_epoch_loss=0, avg_epoch_loss=[], epoch=0)

        # Either provide model_path or use load_checkpoint()
        self.model = ADNet(n_actions=params.num_actions, n_action_history=params.num_action_history)
        if model_path is not None:
            self.model.load_network(model_path, freeze_backbone=False)
            self.model.to(self.device)

        # Set different learning rates for backbone and FC layers.
        self.optimizer = torch.optim.SGD([
            {'params': self.model.backbone.parameters(), 'lr': params.lr_backbone},
            {'params': self.model.fc4_5.parameters()},
            {'params': self.model.fc6.parameters()},
            {'params': self.model.fc7.parameters()}],
            lr=params.lr_train, momentum=params.momentum, weight_decay=params.weight_decay)

        self.decay_learning_rate = False
        if self.decay_learning_rate:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.85)

    def train_epoch(self, train_db, test_db):
        pass

    def train(self, train_db, test_db=[]):
        if self.stats['epoch'] >= self.n_epochs:
            return self.model

        while self.stats['epoch'] < self.n_epochs:
            print("=" * 40, "Starting epoch %d" % (self.stats['epoch'] + 1), "=" * 40)
            self.stats['epoch'] += 1
            self.stats['total_epoch_loss'] = 0.0

            self.model.train()

            if self.decay_learning_rate:
                self.scheduler.step()

            self.train_epoch(train_db, test_db)

            self.stats['avg_epoch_loss'].append(self.stats['total_epoch_loss'] / len(train_db))

            if self.stats['epoch'] % self.epoch_checkpoint == 0:
                self.save_checkpoint()
            if self.evaluate_performance_:
                self.evaluate_performance(train_db, test_db)
            self.print_stats()

        # Save final model
        self.save_checkpoint()

        return self.model

    def evaluate_performance(self, train_db, test_db):
        pass

    def print_stats(self):
        print(self.stats)

    def load_checkpoint(self, path=None):
        if path is None:
            # Resume latest checkpoint
            path = latest_checkpoint(self.checkpoint_path)
            if path is None:
                return False

        print(path)
        state = torch.load(path)
        
        self.stats = state['stats']

        self.model.load_state_dict(state['model'])
        self.model.backbone.requires_grad_(True)
        self.model.to(self.device)

        self.optimizer.load_state_dict(state['optimizer'])
        
        if 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])

        return True

    def save_checkpoint(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path / f"checkpoint_{self.stats['epoch']}.pth"

        state = {
            'stats': self.stats,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            # scheduler
        }
        torch.save(state, path)
        print(path)


class TrainTracker_SL(TrackerTrainer):
    def __init__(self, params, model_path=None, **kwargs):
        super().__init__(params, model_path, **kwargs)

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.model.backbone.parameters(), 'lr': params.lr_backbone},
        #     {'params': self.model.fc4_5.parameters()},
        #     {'params': self.model.fc6.parameters()},
        #     {'params': self.model.fc7.parameters()}],
        #     lr=params.lr_train)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.n_epochs = kwargs.get('epochs', params.n_epochs_sl)
        self.epoch_checkpoint = kwargs.get('epoch_checkpoint', params.checkpoint_interval_sl)

        self.stats['train_acc'] = 0
        self.stats['test_acc'] = 0

    def train_epoch(self, train_db, test_db):
        db_idx = np.random.permutation(len(train_db))  # Shuffle sequences
        for i, seq_idx in enumerate(tqdm(db_idx)):
            seq = train_db[int(seq_idx)]
            for pos_data, neg_data in SequenceDataset(seq, self.params):
            # for pos_data, neg_data in DataLoader(SequenceDataset(seq, self.params), shuffle=True, batch_size=self.params.batch_frames, pin_memory=True, collate_fn=SequenceDataset._collate, num_workers=1):
                pos_patches, pos_actions, pos_scores = pos_data
                neg_patches, neg_scores = neg_data
                
                pos_patches = pos_patches.to(self.device) 
                pos_actions = pos_actions.to(self.device)
                pos_scores = pos_scores.to(self.device)

                neg_patches = neg_patches.to(self.device)
                neg_scores = neg_scores.to(self.device)

                # Action history is set to zero in SL training!
                action_history_oh_zero = torch.tensor(0.0).expand(pos_patches.size(0), self.model.action_history_size).to(self.device)

                # TODO why not combine positive and negatives?
                
                # Optimize for positive samples
                self.optimizer.zero_grad()
                out_actions, out_scores = self.model(pos_patches, action_history_oh_zero, log_softmax=True)
                action_loss = self.criterion(out_actions, pos_actions)
                score_loss = self.criterion(out_scores, pos_scores)
                loss1 = action_loss + score_loss  # Loss sum from paper
                loss1.backward()
                self.optimizer.step()

                action_history_oh_zero = torch.tensor(0.0).expand(neg_patches.size(0), self.model.action_history_size).to(self.device)

                # Optimize for negative samples
                # In this case we don't have any action labels, so don't optimize for action_loss.
                self.optimizer.zero_grad()
                _, out_scores = self.model(neg_patches, action_history_oh_zero, log_softmax=True)
                loss2 = self.criterion(out_scores, neg_scores)
                loss2.backward()
                self.optimizer.step()

                self.stats['total_epoch_loss'] += loss1.item() + loss2.item()

    def evaluate_performance(self, train_db, test_db):
        train_acc = classification_accuracy(self.model, train_db, self.device, self.params)
        test_acc = classification_accuracy(self.model, test_db, self.device, self.params)
        self.stats['train_acc'] = train_acc
        self.stats['test_acc'] = test_acc

    def print_stats(self):
        # super().print_stats()
        print(
            colorama.Fore.GREEN
            + f"\nEpoch {self.stats['epoch']}/{self.n_epochs}, Loss={self.stats['avg_epoch_loss'][-1]:.4f}, Train-Acc={self.stats['train_acc']}, Valid-Acc={self.stats['test_acc']}",
            colorama.Fore.RESET
        )


class TrainTracker_RL(TrackerTrainer):
    REWARD_VERSIONS = ["PAPER", "MATLAB", "RETURNS", "ALL_OVERLAPS", "ALL_REWARDS"]

    def __init__(self, params, reward_version="RETURNS", **kwargs):
        super().__init__(params, **kwargs)
        
        self.n_epochs = kwargs.get('epochs', params.n_epochs_rl)
        self.epoch_checkpoint = kwargs.get('epoch_checkpoint', params.checkpoint_interval_rl)

        self.stats['reward_version'] = reward_version
        self.stats['rewards'] = []
        self.stats['actions'] = []
        self.stats['avg_epoch_reward'] = []

    def extract_region(self, image, bbox):
        return extract_region(image, bbox, crop_size=self.params.img_size, 
                                           padding=self.params.padding, 
                                           means=self.params.means)

    def track_frame(self, image, bbox, action_history_oh):
        # Perform actions until convergence in current frame (for RL).
        params = self.params 
        opts = params.opts

        # self.model.eval()

        img_size = np.array(image.shape[1::-1])

        log_probs = []
        bboxes = [bbox]
        
        # For oscillation checking
        round_bboxes = set()
        round_bboxes.add(tuple(bbox.round()))

        move_counter = 0
        prev_action = -1
        curr_bbox = bbox
        while move_counter < opts['num_action_step_max'] and prev_action != opts['stop_action']:
            curr_patch = self.extract_region(image, curr_bbox).to(self.device)

            actions, conf = self.model(curr_patch, action_history_oh, log_softmax=False)
            prob_actions = Categorical(probs=actions)  # For action sampling

            # NOTE more correct to always sample actions instead of taking the max (like in matlab code)
            # However in evaluation tracking, I take just the max.
            action = prob_actions.sample()
            action_idx = action.item()
            log_probs.append(prob_actions.log_prob(action))
            self.stats['actions'].append(action_idx)

            next_bbox = perform_action(curr_bbox, action_idx, params)
            next_bbox = clamp_bbox(next_bbox, img_size)

            # Check for oscillations
            next_bbox_round = tuple(next_bbox.round())
            if move_counter > 0 and action_idx != opts['stop_action'] and next_bbox_round in round_bboxes:
                action_idx = opts['stop_action']  # Don't store this action because it wasn't picked by the model
                self.stats['actions'].append(action_idx)

            # Update one-hot action history
            n = self.params.num_actions
            action_history_oh[0, n:] = action_history_oh[0, :-n]
            action_history_oh[0, :n] = one_hot(torch.tensor(action_idx), num_classes=n)

            curr_bbox = next_bbox
            bboxes.append(curr_bbox)
            round_bboxes.add(next_bbox_round)
            prev_action = action_idx
            move_counter += 1

        # self.stats['move_counter'] = move_counter
        return np.stack(bboxes), log_probs

    def calc_weights(self, sim_overlaps, version):
        if version == "PAPER":
            # The reward z_t,l for each action is computed only from the terminal state! (in paper)
            weights = []
            for overs in sim_overlaps:
                reward = 1 if overs[-1] >= 0.7 else -1  # Final box overlap
                weights.append(reward * torch.ones(len(overs)))
            weights = torch.cat(weights)
        elif version == "MATLAB":
            # In matlab code, the reward is computed for the whole sequence from just the final (gt, bbox) pair!
            # ADNet matlab version: reward whole sequence based on final frame IoU
            n = sum(len(x) for x in sim_overlaps)
            reward = 1 if sim_overlaps[-1][-1] >= 0.7 else -1
            weights = reward * torch.ones(n)
        elif version == "RETURNS":
            # Calculate rewards-to-go
            rewards = np.concatenate(sim_overlaps)
            R = 0
            weights = []
            for r in rewards[::-1]:
                R = r + self.params.rl_gamma * R
                weights.append(R)
            weights = torch.tensor(weights[::-1])
            weights = (weights - weights.mean()) / (weights.std() + 1e-9)  # normalize discounted rewards
        elif version == "ALL_OVERLAPS":
            # reward == overlap
            weights = torch.from_numpy(np.concatenate(sim_overlaps)).float()
        elif version == "ALL_REWARDS":
            # 1, -1 based on overlap
            weights = torch.from_numpy(np.concatenate(sim_overlaps)).float()
            idx = weights >= 0.7
            weights[idx] = 1
            weights[~idx] = -1
        return weights

    def train_epoch(self, train_db, test_db):
        self.model.train()

        total_epoch_reward = 0

        db_idx = np.random.permutation(len(train_db))  # Shuffle sequences
        # db_idx = np.arange(len(train_db))
        for i, seq_idx in enumerate(tqdm(db_idx)):
            seq = train_db[int(seq_idx)]

            # Collect experience by acting in the environment with current policy
            # Batching decreases variance
            batch_log_probs = []
            batch_weights = []
            batch_rewards = []

            # All simulations in the batch are sampled from the same sequence (correct?)
            sampler = RandomSequenceSampler([seq], self.params.rl_sequence_length, samples_per_epoch=self.params.n_batches_rl)
            for sequence, in DataLoader(sampler, batch_size=1, num_workers=0, collate_fn=identity_func):
            # for _ in range(self.params.n_batches_rl):
                # sequence = self.sequence_sampler[i]  # Random sequence of fixed length (10)
                sequence = sampler[0]
                # self.stats['sequences'].append(str(sequence))

                sim_log_probs = []
                sim_overlaps = []
                sim_total_reward = 0  # Not actually the 'reward' but IoU overlap!
                
                # Simulate sequence and gather results
                curr_bbox = sequence.ground_truth_rect[0]
                image = image_loader(sequence.frames[0])
                action_history_oh = torch.zeros(1, self.model.action_history_size).to(self.device)  # Updated in-place
                for frame_num, frame_path in enumerate(sequence.frames[1:], start=1):
                    image = image_loader(frame_path)
                    bboxes, log_probs = self.track_frame(image, curr_bbox, action_history_oh)

                    # Compute "rewards" for each action used in simulation.
                    next_bbox = bboxes[-1]
                    gt = sequence.ground_truth_rect[frame_num]
                    frame_overlaps = overlap_ratio(gt, bboxes[1:])

                    sim_total_reward += frame_overlaps[-1]  # Final state overlap
                    sim_overlaps.append(frame_overlaps)
                    sim_log_probs.extend(log_probs)

                    curr_bbox = next_bbox

                sim_weights = self.calc_weights(sim_overlaps, version=self.stats['reward_version'])
                sim_log_probs = torch.cat(sim_log_probs)

                batch_log_probs.append(sim_log_probs)
                batch_weights.append(sim_weights)

                # NOTE don't keep track of total reward by summing rewards because longer actions would yield more reward...
                sequence_reward = sim_total_reward / (len(sequence.frames) - 1)
                batch_rewards.append(sequence_reward)

            batch_log_probs = torch.cat(batch_log_probs).to(self.device)
            batch_weights = torch.cat(batch_weights).to(self.device)

            # Policy gradient (loss)
            # NOTE: scores (fc7) are ignored during RL training
            # -1 for maximization using minimization algorithm
            self.optimizer.zero_grad()
            loss = -(batch_log_probs * batch_weights).mean()
            loss.backward()
            self.optimizer.step()

            # TODO we could also train fc7 here ...

            batch_reward = sum(batch_rewards) / len(batch_rewards)
            total_epoch_reward += batch_reward
            self.stats['rewards'].append(batch_reward)
            self.stats['total_epoch_loss'] += loss.item()
        self.stats['avg_epoch_reward'].append(total_epoch_reward / len(train_db))

    def evaluate_performance(self, train_db, test_db):
        pass

    def print_stats(self):
        # super().print_stats()
        print(
            colorama.Fore.GREEN
            + f"\nEpoch {self.stats['epoch']}/{self.n_epochs}, Loss={self.stats['avg_epoch_loss'][-1]:.4f}, Reward={self.stats['avg_epoch_reward'][-1]:.4f}",
            colorama.Fore.RESET
        )
