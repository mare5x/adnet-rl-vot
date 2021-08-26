import random
import pathlib
import numpy as np
import torch
from torch.nn.functional import one_hot
from pytracking.tracker.base import BaseTracker

from .models import ADNet
from .utils import gen_samples, overlap_ratio, extract_region, RegionExtractor, SampleGenerator, PIL_plot_image
from .train import generate_action_labels, perform_action


# Helper class for storing positive/negative online training samples.
class OnlineDataset:
    def __init__(self, type_, max_frames):
        self.type = type_
        self.max_frames = max_frames
        self.data = []
    
    def add(self, features, action_labels=None):
        self.data.append((features, action_labels))
        if len(self.data) > self.max_frames:
            del self.data[0]

    def get(self, n=None):
        n = len(self.data) if n is None else min(n, len(self.data))
        feats = torch.cat([d[0] for d in self.data[-n:]], dim=0)
        if self.type == 'positive':
            action_labels = torch.cat([d[1] for d in self.data[-n:]], dim=0)
            score_labels = torch.tensor(1).expand(feats.size(0))
        else:
            action_labels = None
            score_labels = torch.tensor(0).expand(feats.size(0))
        return feats, action_labels, score_labels


# Port adnet_test.m 
# This class is for evalutaion only (training is separate!)
class ADNetTracker(BaseTracker):
    
    multiobj_mode = 'parallel'

    def initialize_features(self): pass

    def initialize(self, image, info):
        self.debug_info = {}
        self.frame_num = 1

        self.img_size = np.array(image.shape[1::-1])

        opts = self.params.opts

        # Use GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init network
        self.adnet = ADNet(n_actions=self.params.num_actions, n_action_history=self.params.num_action_history)
        self.adnet.load_network(self.params.model_path, freeze_backbone=True)  # Also freezes backbone
        self.adnet.to(self.device)

        # For preprocessing
        self.pos_generator = SampleGenerator('gaussian', self.img_size, self.params.trans_pos, self.params.scale_pos)
        self.neg_generator = SampleGenerator('uniform', self.img_size, self.params.trans_neg, self.params.scale_neg) 

        self.pos_dataset = OnlineDataset('positive', opts['nFrames_long'])
        self.neg_dataset = OnlineDataset('negative', opts['nFrames_short'])

        # Optimizer and criterion used in the paper
        self.optimizer = torch.optim.SGD(self.adnet.parameters(), 
            lr=self.params.lr_update, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

        # self.batch_size = 1  # Cannot batch when evaluating a sequence ...

        # One hot encoded action history ("action dynamics vector d") is zero initially (1x110)
        self.action_history_oh = torch.zeros(1, self.params.num_actions * self.params.num_action_history).to(self.device)
        self.action_history = [] 

        box = np.array(info['init_bbox'], dtype="float32")

        self.prev_bbox = box
        self.prev_action_history_oh = self.action_history_oh.detach().clone()
        self.cont_negatives = 0  # How many continuous negative frames?

        # Perform initial frame training
        if self.params.initial_fine_tuning:
            self.initial_fine_tuning(image, box)

        if self.params.save_visualization:
            self.params.visualization_path.mkdir(exist_ok=True)
            fig = PIL_plot_image(image, gt=box)
            fig.save(self.params.visualization_path / f"{self.frame_num:06}.jpg")

    def initial_fine_tuning(self, image, bbox):
        pos_samples = self.pos_generator(bbox, self.params.n_pos_init, self.params.overlap_pos_init)
        
        # Note the different sampling strategy for negative samples
        n_neg_half = int(self.params.n_neg_init * 0.5)
        neg_samples = np.concatenate([
            SampleGenerator('uniform', self.img_size, self.params.trans_neg_init, self.params.scale_neg_init)(
                bbox, n_neg_half, self.params.overlap_neg_init),
            SampleGenerator('whole', self.img_size)(
                bbox, n_neg_half, self.params.overlap_neg_init)])
        neg_samples = np.random.permutation(neg_samples)

        pos_feats, neg_feats = self.extract_features(image, pos_samples, neg_samples)
        pos_action_labels = torch.from_numpy(self.generate_action_labels(bbox, pos_samples))

        self.pos_dataset.add(pos_feats, pos_action_labels)
        self.neg_dataset.add(neg_feats)  # TODO MDNet instead generates a new batch of negatives ...

        # Fine tune, but first change optimizer (ADNet changes learning rate to 0.0003)
        original_optimizer = self.optimizer
        self.optimizer = torch.optim.SGD(self.adnet.parameters(), 
            lr=self.params.lr_init, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        self.online_fine_tuning(self.pos_dataset.get(), self.neg_dataset.get(), self.params.maxiter_init)
        self.optimizer = original_optimizer

    def clamp_bbox(self, bbox, img_size):
        bbox[2:] = np.clip(bbox[2:], 10, img_size - 10)
        bbox[:2] = np.clip(bbox[:2], 0, img_size - bbox[2:] - 1)
        return bbox

    def take_action(self, bbox, action_idx):
        return perform_action(bbox, action_idx, self.params)

    def update_action_history(self, action_idx):
        # NOTE: stop actions are included in the action history!

        self.action_history.append(action_idx)
        
        # FIFO style array
        n = self.params.num_actions
        self.action_history_oh[0, n:] = self.action_history_oh[0, :-n]
        self.action_history_oh[0, :n] = one_hot(torch.tensor(action_idx), num_classes=n)

    def is_inverse_action(self, action_prev, action_curr):
        return action_curr == self.params.inverse_actions[action_prev]

    def tracking_procedure(self, image, bbox):
        # Perform actions until convergence in current frame.
        # From supplementary material.
        
        self.adnet.eval()

        opts = self.params.opts
        bboxes = [bbox]
        scores = []
        move_counter = 0
        prev_action = -1

        # For oscillation checking
        round_bboxes = set()
        round_bboxes.add(tuple(bbox.round()))

        curr_bbox = bbox
        curr_score = 0
        is_negative = False
        while move_counter < opts['num_action_step_max'] and prev_action != opts['stop_action']:
            curr_patch = self.extract_region(image, curr_bbox).to(self.device)

            with torch.no_grad():
                actions, conf = self.adnet(curr_patch, self.action_history_oh)

            curr_score = conf[0, 1].item()  # Batch 0, second output neuron
            scores.append(curr_score)
            # Stop tracking if not confident  (TODO really?)
            is_negative = curr_score < opts['failedThre']
            if is_negative and self.params.stop_unconfident:
                self.cont_negatives += 1
                # curr_score = prev_score
                self.update_action_history(opts['stop_action'])
                break

            action_idx = actions.argmax(dim=1).item()

            next_bbox = self.take_action(curr_bbox, action_idx)
            next_bbox = self.clamp_bbox(next_bbox, self.img_size)
            
            self.update_action_history(action_idx)

            # Check for oscillations (important for example: SHRINK -> EXPAND -> SHRINK ...)
            # 2 options: (1) check for opposite action or (2) check for existing bounding box
            # Option (2) is better: LEFT2 -> RIGHT -> RIGHT is an oscillation
            next_bbox_round = tuple(next_bbox.round())
            if move_counter > 0 and action_idx != opts['stop_action'] and next_bbox_round in round_bboxes:
                action_idx = opts['stop_action']
                self.update_action_history(action_idx)  # TODO pop last 2 actions?

            curr_bbox = next_bbox
            bboxes.append(curr_bbox)
            round_bboxes.add(next_bbox_round)
            prev_action = action_idx
            move_counter += 1

        self.debug_info['move_counter'] = move_counter
        if self.params.debug >= 3 or self.params.save_visualization: 
            self.debug_info['action_bboxes'] = bboxes
        self.debug_info['scores'] = scores
        self.debug_info['action_history'] = self.action_history[-move_counter - 1:]
        return curr_bbox, is_negative, curr_score

    def redetect(self, image, bbox, confidence):
        batch_size = self.params.opts['redet_samples']
        trans_f = min(1.5, 0.6 * pow(1.15, self.cont_negatives))
        samples = gen_samples('gaussian', bbox, batch_size, self.img_size, trans_f, self.params.opts['redet_scale_factor'])
        
        # Batch all patches and expand action history
        patches = self.extract_regions(image, samples).to(self.device)
        # TODO matlab code uses old action history
        action_histories_oh = self.action_history_oh.expand(batch_size, -1).to(self.device)

        self.adnet.eval()
        with torch.no_grad():
            _, confs = self.adnet(patches, action_histories_oh)

        # Paper version: pick patch with highest confidence
        # best_idx = confs[:, 1].argmax(dim=0).item()
        # new_bbox = samples[best_idx, :]
        # return new_bbox

        # Matlab code version: take the mean if better
        values, indices = confs[:, 1].topk(5)
        mean_score = values.mean()
        if mean_score > confidence:
            return samples[indices.cpu()].mean(axis=0), mean_score
        return bbox, confidence

    def extract_region(self, image, bbox):
        return extract_region(image, bbox, crop_size=self.params.img_size, 
                                           padding=self.params.padding, 
                                           means=self.params.means)

    def extract_regions(self, image, samples):
        return RegionExtractor(image, samples, self.params)()

    def extract_features(self, image, *samples):
        # Extract features from bbox samples.
        self.adnet.eval()

        sizes = [s.shape[0] for s in samples]
        all_samples = np.concatenate(samples)
        extractor = RegionExtractor(image, all_samples, self.params)
        for i, regions in enumerate(extractor):  # Batched iteration
            regions = regions.to(self.device)
            with torch.no_grad():
                feat = self.adnet.extract_features(regions)
            if i==0:
                feats = feat.detach().clone()
            else:
                feats = torch.cat((feats, feat.detach().clone()), dim=0)
        return feats.split(sizes)

    def generate_box_samples(self, bbox):
        # Sample positive and negative patches around the given bounding box.
        pos_examples = self.pos_generator(bbox, self.params.n_pos_update, overlap_range=self.params.overlap_pos)
        neg_examples = self.neg_generator(bbox, self.params.n_neg_update, overlap_range=self.params.overlap_neg)
        return pos_examples, neg_examples

    def generate_action_labels(self, bbox, samples):
        return generate_action_labels(bbox, samples, self.params)

    def generate_learning_samples(self, image, bbox):
        pos_samples, neg_samples = self.generate_box_samples(bbox)

        # Extract features (smaller size for storage)
        pos_feats, neg_feats = self.extract_features(image, pos_samples, neg_samples)

        # Action labels only for positive samples
        action_labels = self.generate_action_labels(bbox, pos_samples)
        action_labels = torch.from_numpy(action_labels)

        # Score/confidence labels are implicit: 1 for positive samples, 0 for negative samples
        return (pos_feats, action_labels), neg_feats

    def online_fine_tuning(self, pos_data, neg_data, maxiter):
        self.adnet.train()

        opts = self.params.opts
        batch_size = opts['minibatch_size']
        batch_test = self.params.batch_test  # Batch size during evaluation in hard mining
        batch_neg_cand = max(self.params.batch_neg_cand, batch_size)
        if not self.params.hard_negative_mining:
            batch_neg_cand = batch_size

        pos_feats, pos_actions, pos_scores = pos_data
        neg_feats, neg_actions, neg_scores = neg_data

        # Action history is set to zero in SL training!
        action_history_oh_zero = torch.tensor(0.0).expand(batch_size, self.action_history_oh.size(1)).to(self.device)

        # Shuffled index vectors for positive/negative samples
        pos_idx = torch.randperm(pos_feats.size(0))
        neg_idx = torch.randperm(neg_feats.size(0))
        while(len(pos_idx) < batch_size * maxiter):
            pos_idx = torch.cat([pos_idx, torch.randperm(pos_feats.size(0))], dim=0)
        while(len(neg_idx) < batch_neg_cand * maxiter):
            neg_idx = torch.cat([neg_idx, torch.randperm(neg_feats.size(0))], dim=0)

        pos_pointer = 0
        neg_pointer = 0

        for it in range(maxiter):
            pos_cur_idx = pos_idx[pos_pointer:(pos_pointer + batch_size)]
            neg_cur_idx = neg_idx[neg_pointer:(neg_pointer + batch_neg_cand)]
            pos_pointer += batch_size
            neg_pointer += batch_neg_cand

            # Create current batch
            batch_pos_feats = pos_feats[pos_cur_idx].to(self.device)
            batch_pos_actions = pos_actions[pos_cur_idx].to(self.device)
            batch_pos_scores = pos_scores[pos_cur_idx].to(self.device)  # All 1

            batch_neg_feats = neg_feats[neg_cur_idx].to(self.device)
            batch_neg_scores = neg_scores[neg_cur_idx].to(self.device)  # All 0

            # Hard negative mining
            # Find those negative examples for which the model is most confident (hence it is most incorrect).
            if self.params.hard_negative_mining:
                if batch_neg_cand > batch_size:
                    self.adnet.eval()
                    for start in range(0, batch_neg_cand, batch_test):
                        end = min(start + batch_test, batch_neg_cand)
                        with torch.no_grad():
                            oh = torch.tensor(0.0).expand(end - start, action_history_oh_zero.size(1)).to(self.device)
                            _, score = self.adnet(batch_neg_feats[start:end], oh, skip_backbone=True)
                        if start == 0:
                            neg_cand_score = score.detach()[:, 1].clone()
                        else:
                            neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

                    _, top_idx = neg_cand_score.topk(batch_size)
                    batch_neg_feats = batch_neg_feats[top_idx]
                    batch_neg_scores = batch_neg_scores[top_idx]
                    self.adnet.train()

            # Optimize for positive samples
            self.optimizer.zero_grad()
            out_actions, out_scores = self.adnet(batch_pos_feats, action_history_oh_zero, skip_backbone=True, log_softmax=True)
            action_loss = self.criterion(out_actions, batch_pos_actions)
            score_loss = self.criterion(out_scores, batch_pos_scores)
            loss1 = action_loss + score_loss  # Loss sum from paper
            loss1.backward()
            self.optimizer.step()

            # Optimize for negative samples
            # In this case we don't have any action labels, so don't optimize for action_loss.
            self.optimizer.zero_grad()
            _, out_scores = self.adnet(batch_neg_feats, action_history_oh_zero, skip_backbone=True, log_softmax=True)
            loss2 = self.criterion(out_scores, batch_neg_scores)
            loss2.backward()
            self.optimizer.step()

            if self.params.debug >= 2:
                print(f"Iteration {it}: {loss1:.4f} | {loss2:.4f}")

    def track_debug(self, info):
        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        self.debug_info.update(info)

        if self.visdom is not None:
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

            if self.params.debug >= 3:
                img = info['image'].copy()
                self.visdom.register((img, *[list(b) for b in self.debug_info['action_bboxes']]), 'Tracking', 1, 'Actions')

        if self.params.debug >= 4:
            print(self.debug_info)

    def track(self, image, info: dict = None) -> dict:
        track_info = dict()
        if self.params.debug >= 3: track_info['image'] = image

        # Track
        bbox, is_negative, score = self.tracking_procedure(image, self.prev_bbox)
        
        # Redetection
        self.debug_info['is_negative'] = is_negative
        if is_negative and self.params.redetection:
            # TODO use old or new bbox? If old, use old action history in redetect otherwise use new action history.
            bbox, score = self.redetect(image, bbox, score)
        
        if self.params.fine_tuning:
            # Positive/negative training samples
            if not is_negative or score > self.params.opts['successThre']:
                self.cont_negatives = 0
                
                (pos_feats, pos_actions), neg_feats = self.generate_learning_samples(image, bbox)
                self.pos_dataset.add(pos_feats, pos_actions)
                self.neg_dataset.add(neg_feats)

            # Online fine tuning
            is_finetune_interval = self.frame_num % self.params.opts['finetune_interval'] == 0
            if is_negative:
                pos_data = self.pos_dataset.get(self.params.opts['nFrames_short'])
                neg_data = self.neg_dataset.get()
                self.online_fine_tuning(pos_data, neg_data, self.params.opts['finetune_iters_online'])
            elif is_finetune_interval:
                pos_data = self.pos_dataset.get()
                neg_data = self.neg_dataset.get()
                self.online_fine_tuning(pos_data, neg_data, self.params.opts['finetune_iters_online'])

        if self.params.save_visualization:
            fig = PIL_plot_image(image, self.debug_info['action_bboxes'])
            fig.save(self.params.visualization_path / f"{self.frame_num:06}.jpg")

        # Bookkeeping
        track_info['target_bbox'] = bbox
        track_info['target_score'] = score
        self.prev_bbox = track_info['target_bbox']
        self.prev_action_history_oh = self.action_history_oh.detach().clone()

        self.track_debug(track_info)
        self.debug_info = {}
        out = {'target_bbox': track_info['target_bbox'].tolist()}
        return out
