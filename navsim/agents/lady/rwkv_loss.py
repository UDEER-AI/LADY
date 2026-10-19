from typing import Dict
from scipy.optimize import linear_sum_assignment
import logging

import torch
import torch.nn.functional as F

from navsim.agents.lady.rwkv_config import RWKVConfig
from navsim.agents.lady.rwkv_features import BoundingBox2DIndex

logger = logging.getLogger(__name__)

def RWKV_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    pred_final_pdms_scores,
    target_scores: torch.Tensor,
    final_scores: torch.Tensor,
    best_scores: torch.Tensor,
    gt_states: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_ego_areas: torch.Tensor,
    config: RWKVConfig
):
    """
    Helper function calculating complete loss of RWKV model
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global RWKV config
    :return: combined loss value
    """
    if "best_trajectory" in predictions:
        trajectory_reg_loss = _traj_regression_loss(predictions["best_trajectory"], targets)
    else:
        trajectory_reg_loss = _traj_regression_loss([predictions["trajectory"]], targets)
    if isinstance(target_scores, list):
        total_sub_score_loss = 0.0
        total_final_score_loss = 0.0
        for i in range(len(target_scores)):
            if len(predictions["score_logit2"]) > 0:
                sub_score_loss, final_score_loss = _traj_score_loss(predictions["score_logit"][i], predictions["score_logit2"][i], target_scores[i])
            else:
                sub_score_loss, final_score_loss = _traj_score_loss(predictions["score_logit"][i], None, target_scores[i])
        total_sub_score_loss += sub_score_loss
        total_final_score_loss += final_score_loss
        score_loss = [total_sub_score_loss, total_final_score_loss]
    else:
        score_loss = _traj_score_loss(predictions["score_logit"], predictions["score_logit2"], target_scores)
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )
    weighted_trajectory_loss = config.trajectory_weight * config.trajectory_reg_weight * trajectory_reg_loss
    weighted_trajectory_scorer_loss = config.trajectory_weight * (config.trajectory_sub_score_weight * score_loss[0] + config.trajectory_final_score_weight * score_loss[1])
    weighted_agent_class_loss = config.agent_class_weight * agent_class_loss
    weighted_agent_box_loss = config.agent_box_weight * agent_box_loss
    weighted_bev_semantic_loss = config.bev_semantic_weight * bev_semantic_loss

    loss = weighted_trajectory_loss + weighted_trajectory_scorer_loss + weighted_agent_class_loss + weighted_agent_box_loss + weighted_bev_semantic_loss

    if predictions["pred_agent_col"] is not None:
        pred_states = predictions["pred_agent_col"][..., :-1].reshape(gt_states.shape)
        pred_logits = predictions["pred_agent_col"][..., -1:].reshape(gt_valid.shape)

        pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

        if len(pred_l1_loss):
            pred_l1_loss = pred_l1_loss.mean()
        else:
            pred_l1_loss = pred_states.mean() * 0

        pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")
        weighted_agent_ce_loss = config.agent_ce_weight * pred_ce_loss
        weighted_agent_l1_loss = config.agent_l1_weight * pred_l1_loss
        loss += weighted_agent_ce_loss
        loss += weighted_agent_l1_loss
    else:
        weighted_agent_ce_loss = 0.0
        weighted_agent_l1_loss = 0.0

    if predictions["pred_area"] is not None:
        pred_area_logits = predictions["pred_area"].reshape(gt_ego_areas.shape)

        pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                          reduction="mean")
        weighted_area_loss = config.area_weight * pred_area_loss
        loss += weighted_area_loss
    else:
        weighted_area_loss = 0


    return (
        loss, weighted_trajectory_loss, weighted_trajectory_scorer_loss, 
        weighted_agent_class_loss, weighted_agent_box_loss, 
        weighted_bev_semantic_loss, weighted_agent_ce_loss, weighted_agent_l1_loss, weighted_area_loss, 
        torch.mean(predictions["best_pdm_score"]), torch.mean(predictions["average_pdm_score"]), 
        torch.mean(best_scores), torch.mean(final_scores), torch.mean(pred_final_pdms_scores)
    )


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: RWKVConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global RWKV config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]
    if torch.isnan(pred_states).any() or torch.isnan(pred_logits).any():
        logger.info("NaN in agent states")
        return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()
    if torch.isnan(cost).any():
        logger.info("NaN in cost matrix")
        return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _traj_regression_loss(best_trajectories, targets):
    """
    Compute the loss for the regression and classification of the trajectory.
    Args:
        best_trajectories: [(bs, 20, 8, 3), ...]
        targets: dict, contains the target trajectory (bs, 8, 3)
    """
    reg_loss = 0.0
    target_traj = targets["trajectory"]
    for best_traj in best_trajectories:
        reg_loss += F.l1_loss(best_traj, target_traj)
    return reg_loss


def _traj_score_loss(pred_logit, pred_logit2, target_scores):
    sub_score_loss = F.binary_cross_entropy_with_logits(pred_logit, target_scores[..., -pred_logit.shape[-1]:])  # .mean()[..., -6:]
    final_score_loss = F.binary_cross_entropy_with_logits(pred_logit[..., -1], target_scores[..., -1])  # .mean()
    if pred_logit2 is not None:
        sub_score_loss += F.binary_cross_entropy_with_logits(pred_logit2, target_scores[..., -pred_logit2.shape[-1]:])  # .mean()[..., -6:]
        final_score_loss += F.binary_cross_entropy_with_logits(pred_logit2[..., -1], target_scores[..., -1])  # .mean()
        sub_score_loss /= 2.0
        final_score_loss /= 2.0
    return sub_score_loss, final_score_loss
