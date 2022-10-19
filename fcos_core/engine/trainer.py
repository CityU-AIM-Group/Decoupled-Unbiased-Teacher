# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import itertools

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as TT
from torchvision.utils import save_image
from collections import OrderedDict

from fcos_core.structures.image_list import to_image_list
from fcos_core.structures.bounding_box import BoxList
from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later, synchronize
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.engine.inference import inference

from fcos_core.data import transforms as T

@torch.no_grad()
def _update_teacher_model(generator_t, generator_s, predictor_t, predictor_s, keep_rate=0.996):
    '''
    PTR for teacher update
    '''
    if get_world_size() > 1:
        student_generator_dict = {
            key[7:]: value for key, value in generator_s.state_dict().items()
        }
        student_predictor_dict = {
            key[7:]: value for key, value in predictor_s.state_dict().items()
        }
    else:
        student_generator_dict = generator_s.state_dict()
        student_predictor_dict = predictor_s.state_dict()

    new_teacher_generator_dict = OrderedDict()
    new_teacher_predictor_dict = OrderedDict()
    for key, value in generator_t.state_dict().items():
        if key in student_generator_dict.keys():
            new_teacher_generator_dict[key] = (
                student_generator_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    for key, value in predictor_t.state_dict().items():
        if key in student_predictor_dict.keys():
            new_teacher_predictor_dict[key] = (
                student_predictor_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    generator_t.load_state_dict(new_teacher_generator_dict)
    predictor_t.load_state_dict(new_teacher_predictor_dict)
    return generator_t, predictor_t

@torch.no_grad()
def _decouple_teacher_model(generator_t, generator_s, predictor_t, predictor_s, keep_rate=0.996):
    '''
    PTR for teacher update
    '''
    if get_world_size() > 1:
        student_generator_dict = {
            key[7:]: value for key, value in generator_s.state_dict().items()
        }
        student_predictor_dict = {
            key[7:]: value for key, value in predictor_s.state_dict().items()
        }
    else:
        student_generator_dict = generator_s.state_dict()
        student_predictor_dict = predictor_s.state_dict()

    new_teacher_generator_dict = OrderedDict()
    new_teacher_predictor_dict = OrderedDict()
    for key, value in generator_t.state_dict().items():
        if key in student_generator_dict.keys():
            new_teacher_generator_dict[key] = (
                student_generator_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    for key, value in predictor_t.state_dict().items():
        if key in student_predictor_dict.keys():
            new_teacher_predictor_dict[key] = (student_predictor_dict[key])
        else:
            raise Exception("{} is not found in student model".format(key))

    generator_t.load_state_dict(new_teacher_generator_dict)
    predictor_t.load_state_dict(new_teacher_predictor_dict)
    return generator_t, predictor_t

@torch.no_grad() # Requires no grad
def threshold_bbox(list_of_boxes, th=0.8, ugob='v2'):
    new_list_of_boxes = []
    for box_list in list_of_boxes:
        distance = 0. # init confidence
        # get field
        boxes_pre = box_list.bbox
        scores_pre = box_list.get_field("scores")
        labels_pre = box_list.get_field("labels")
        locations_pre = box_list.get_field("locations")
        levels_pre = box_list.get_field("levels")
        assert scores_pre.shape == levels_pre.shape
        # thresholding
        valid_map = scores_pre > th
        ############### At least one possible output ###############
        if sum(valid_map) == 0: # Inconfident sample - -
            inconfident = True
            valid_map = scores_pre >= max(scores_pre)
            distance = th - max(scores_pre)
        else:
            inconfident = False
        ############################################################
        # indexing
        boxes = boxes_pre[valid_map, :].detach()
        scores = scores_pre[valid_map].detach()
        labels = labels_pre[valid_map].detach()
        locations = locations_pre[valid_map].detach()
        levels = levels_pre[valid_map].detach()
        # make new box list
        bbox = BoxList(boxes, box_list.size, box_list.mode)
        bbox.add_field("scores", scores)
        bbox.add_field("labels", labels)
        bbox.add_field("locations", locations)
        bbox.add_field("levels", levels)
        if inconfident == True:
            bbox.add_field("inconfident", True)
        else:
            bbox.add_field("inconfident", False)
        ####################### For UGOB
        if ugob == 'v1':
            uncertainty = 1 / math.exp(distance)
            bbox.add_field("uncertainty", uncertainty)
        elif ugob == 'v2':
            z = 1
            uncertainty = 1/z * torch.exp(scores)/(torch.exp(scores) + torch.exp(1-scores))
            bbox.add_field("uncertainty", uncertainty)
        elif ugob == None:
            bbox.add_field("uncertainty", torch.tensor(1))
        else:
            raise NotImplementedError
        ##################################
        new_list_of_boxes.append(bbox)
    return new_list_of_boxes


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def mixup(img, img2, lam):
    img = lam * img + (1 - lam) * img2
    return img

def do_train(
    cfg,
    model,
    data_loader,
    data_loaders_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    stage,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    for k in model:
        model[k].train()
    logger.info(model)
    if stage != 1:
        generator_t = model["generator_t"]
        generator_t.eval()
        predictor_t = model["predictor_t"]
        predictor_t.eval()
    generator_s = model["generator_s"]
    predictor_s = model["predictor_s"]
    # print(generator_s)
    # print(predictor_s)
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST
    update_part = cfg.SOLVER.SFDA_UPDATE_PART

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    # For saving purpose
    ap = 0 
    ap50 = 0 
    ####################
    k = 0
    igus_filterout_iter = 0
    object_count = 0
    if stage == 1: # Train the network with source data
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            images = images.to(device)
            images = to_image_list(images)
            targets = [target.to(device) for target in targets]

            features = generator_s(images.tensors)
            _, loss_dict_1 = predictor_s(images, features, targets=targets)
            loss_dict_1 = {k: loss_dict_1[k] for k in loss_dict_1}

            losses = sum(loss for loss in loss_dict_1.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced_1 = reduce_loss_dict(loss_dict_1)
            losses_reduced_1 = sum(loss for loss in loss_dict_reduced_1.values())
            meters.update(loss=losses_reduced_1, **loss_dict_reduced_1)
            
            optimizer["generator_s"].zero_grad()
            optimizer["predictor_s"].zero_grad()
            losses.backward()
            for k in optimizer:
                optimizer["generator_s"].step()
                optimizer["predictor_s"].step()

            if pytorch_1_1_0_or_later:
                scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr_G: {lr_G:.6f}",
                            "lr_P1: {lr_P1:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_G=optimizer["generator_s"].param_groups[0]["lr"],
                        lr_P1=optimizer["predictor_s"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if data_loaders_val is not None and test_period > 0 and iteration % test_period == 0:
                synchronize()
                for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
                    results, _ = inference(
                        model,
                        data_loader_val,
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                    )
                ap_tmp, ap50_tmp = results.results['bbox']['AP'], results.results['bbox']['AP50']
                if ap50_tmp > ap50:
                    ap50 = ap50_tmp
                    checkpointer.save("model_best_ap50", **arguments)
                    logger.info("Updating Best mAP50 Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                if ap_tmp > ap:
                    ap = ap_tmp
                    checkpointer.save("model_best_ap", **arguments)
                    logger.info("Updating Best mAP Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                synchronize()
                for k in model:
                    model[k].train()
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )

    else: # Strong-weak
        random_bank = []
        inconfident_bank = []
        global_prototype = torch.rand(256)
        for iteration, (images_weak, images_strong, img_id) in enumerate(data_loader, start_iter):
            # print(img_id) 
            # # ({'file_name': '61b59dd5-871d-4494-a316-1d9477611775.jpg', 'height': 576, 'id': 1327, 'width': 720}, { ... })
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                if update_part == "both":
                    scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            images_weak, images_strong = images_weak.to(device), images_strong.to(device)
            images_weak, images_strong = to_image_list(images_weak), to_image_list(images_strong)

            ####################### TEACHER MODEL #############################
            features_t = generator_t(images_weak.tensors)
            boxes, _ = predictor_t(images_weak, features_t, targets=None)

            # CORE FOR PSEUDO LABELING
            boxes = threshold_bbox(boxes, th=cfg.SOLVER.SFDA_PSEUDO_TH, ugob=cfg.SOLVER.UGOB)
            if cfg.SOLVER.IUGS:
                if iteration % test_period != 0: # the test period let the image pass
                    score0 = boxes[0].get_field("scores")
                    score1 = boxes[1].get_field("scores")
                    scores_all = (score0, score1)
                    # print(scores_all[0].shape)
                    # print(scores_all[1].shape)
                    scores_all = torch.cat(scores_all, 0)
                    scores_all = torch.mean(scores_all)
                    # if scores_all < cfg.SOLVER.IUGS_TH:
                    #     igus_filterout_iter += 1
                    #     logger.info("Filtered out by IGUS! Total times {}/{}".format(igus_filterout_iter, iteration))
                    #     continue
            ###################################################################

            

            ####################### STUDENT MODEL #############################
            features_s = generator_s(images_strong.tensors)

            # CODE FOR FEATURE INTERVAL
            features_t_aux = generator_t(images_strong.tensors)
            if cfg.SOLVER.FEATURE_INTERVAL == True:
                
                features_s = list(features_s)

                features_s, features_t_aux = feature_interval(features_s, features_t_aux)

            student_logit, loss_dict = predictor_s(images_strong, features_s, aux=None, targets=boxes)
            ###################################################################

            if not cfg.SOLVER.SFDA_REG_ON:
                if "loss_reg" in loss_dict:
                    loss_dict.pop("loss_reg")
            if not cfg.SOLVER.SFDA_CTR_ON:
                if "loss_centerness" in loss_dict:
                    loss_dict.pop("loss_centerness")


            loss_dict = {k + "_student": loss_dict[k] for k in loss_dict}

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            
            if update_part == "both":
                optimizer["generator_s"].zero_grad()
            optimizer["predictor_s"].zero_grad()
            losses.backward()

            if update_part == "both":
                optimizer["generator_s"].step()
            optimizer["predictor_s"].step()

            if pytorch_1_1_0_or_later:
                if update_part == "both":
                    scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            if cfg.SOLVER.DIA_ON:
                for idx in range(len(boxes)):
                    if cfg.SOLVER.RANDOM_BANK_ON:
                        # dir = "./outputs/ANALYSIS_VISUALIZE_DIA/mixup/{}.jpg".format(k)
                        add_to_ir(random_bank, images_weak, images_strong, boxes, idx)
                        # k+=1
                        if cfg.SOLVER.DIA_ON:
                            # lam = np.random.beta(0.5, 0.5)
                            dia_mixup_min = cfg.SOLVER.DIA_MIXUP_MIN
                            lam = random.uniform(dia_mixup_min, dia_mixup_min + 0.2)
                            if cfg.SOLVER.RANDOM_BANK_ON:
                                mixer_img = random.sample(random_bank, 1)[0] # sample an inconfident sample
                            else:
                                mixer_img = random.sample(inconfident_bank, 1)[0] # sample an inconfident sample
                            h, w = images_weak.tensors[idx].shape[1], images_weak.tensors[idx].shape[2]
                            images_weak.tensors[idx] = mixup(images_weak.tensors[idx], F.interpolate(mixer_img[0].unsqueeze(0), size=[h, w], mode='bilinear').squeeze().to(device), lam)
                            images_strong.tensors[idx] = mixup(images_strong.tensors[idx], F.interpolate(mixer_img[1].unsqueeze(0), size=[h, w], mode='bilinear').squeeze().to(device), lam)

                        
                        images_weak, images_strong = images_weak.to(device), images_strong.to(device)
                        images_weak, images_strong = to_image_list(images_weak), to_image_list(images_strong)
                        ####################### TEACHER MODEL #############################
                        features_t = generator_t(images_weak.tensors)
                        boxes, _ = predictor_t(images_weak, features_t, targets=None)
                        # CORE FOR PSEUDO LABELING
                        boxes = threshold_bbox(boxes, th=cfg.SOLVER.SFDA_PSEUDO_TH)
                        ###################################################################
                        ####################### STUDENT MODEL #############################
                        features_s = generator_s(images_strong.tensors)

                        # CODE FOR FEATURE INTERVAL
                        features_t_aux = generator_t(images_strong.tensors)
                        if cfg.SOLVER.FEATURE_INTERVAL == True:
                            
                            features_s = list(features_s)
                            features_s, features_t_aux = feature_interval(features_s, features_t_aux)

                        student_logit, loss_dict = predictor_s(images_strong, features_s, aux=None, targets=boxes)
                        ###################################################################

                        if not cfg.SOLVER.SFDA_REG_ON:
                            if "loss_reg" in loss_dict:
                                loss_dict.pop("loss_reg")
                        if not cfg.SOLVER.SFDA_CTR_ON:
                            if "loss_centerness" in loss_dict:
                                loss_dict.pop("loss_centerness")

                        loss_dict = {k + "_student": loss_dict[k] for k in loss_dict}
                        loss_dict_all = loss_dict

                        losses = sum(loss for loss in loss_dict_all.values())

                        # reduce losses over all GPUs for logging purposes
                        loss_dict_reduced = reduce_loss_dict(loss_dict_all)
                        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                        meters.update(loss=losses_reduced, **loss_dict_reduced)
                        
                        if update_part == "both":
                            optimizer["generator_s"].zero_grad()
                        optimizer["predictor_s"].zero_grad()
                        losses.backward()

                        if update_part == "both":
                            optimizer["generator_s"].step()
                        optimizer["predictor_s"].step()

                        if pytorch_1_1_0_or_later:
                            if update_part == "both":
                                scheduler["generator_s"].step()
                            scheduler["predictor_s"].step()
                            
                    else:
                        if boxes[idx].get_field("inconfident") == True: # Store inconfident samples in the bank
                            idx, ins_feats = get_ins_feats(cfg, features_t, boxes, idx)
                            for i in ins_feats:
                                global_prototype = global_prototype.to(i.device)
                                cos = float(torch.cosine_similarity(global_prototype, i, 0))
                                if cos < cfg.SOLVER.COS_TH:
                                    add_to_ir(inconfident_bank, images_weak, images_strong, boxes, idx)
                                    break
                        elif boxes[idx].get_field("inconfident") == False and len(inconfident_bank) != 0: # Regularize the confidence sample with mixup samples
                            if cfg.SOLVER.DIA_SEM_ONLY:
                                idx, ins_feats = get_ins_feats(cfg, features_t, boxes, idx)
                                for i in ins_feats:
                                    global_prototype = global_prototype.to(i.device)
                                    cos = float(torch.cosine_similarity(global_prototype, i, 0))
                                    if cos < cfg.SOLVER.COS_TH:
                                        add_to_ir(inconfident_bank, images_weak, images_strong, boxes, idx)
                                        break
                            ##########################
                            idx, ins_feats = get_ins_feats(cfg, features_t, boxes, idx)
                            for i in ins_feats:
                                global_prototype = global_prototype.to(i.device)
                                momentum = float(torch.cosine_similarity(global_prototype, i, 0))
                                global_prototype = global_prototype * momentum + i * (1 - momentum)
                            ##########################
                            if cfg.SOLVER.DIA_ON:
                                # lam = np.random.beta(0.5, 0.5)
                                dia_mixup_min = cfg.SOLVER.DIA_MIXUP_MIN
                                lam = random.uniform(dia_mixup_min, dia_mixup_min + 0.2)
                                if cfg.SOLVER.RANDOM_BANK_ON:
                                    mixer_img = random.sample(random_bank, 1)[0] # sample an inconfident sample
                                else:
                                    mixer_img = random.sample(inconfident_bank, 1)[0] # sample an inconfident sample
                                h, w = images_weak.tensors[idx].shape[1], images_weak.tensors[idx].shape[2]
                                images_weak.tensors[idx] = mixup(images_weak.tensors[idx], F.interpolate(mixer_img[0].unsqueeze(0), size=[h, w], mode='bilinear').squeeze().to(device), lam)
                                images_strong.tensors[idx] = mixup(images_strong.tensors[idx], F.interpolate(mixer_img[1].unsqueeze(0), size=[h, w], mode='bilinear').squeeze().to(device), lam)

                            
                            images_weak, images_strong = images_weak.to(device), images_strong.to(device)
                            images_weak, images_strong = to_image_list(images_weak), to_image_list(images_strong)
                            ####################### TEACHER MODEL #############################
                            features_t = generator_t(images_weak.tensors)
                            boxes, _ = predictor_t(images_weak, features_t, targets=None)
                            # CORE FOR PSEUDO LABELING
                            boxes = threshold_bbox(boxes, th=cfg.SOLVER.SFDA_PSEUDO_TH)
                            ###################################################################
                            ####################### STUDENT MODEL #############################
                            features_s = generator_s(images_strong.tensors)

                            # CODE FOR FEATURE INTERVAL
                            features_t_aux = generator_t(images_strong.tensors)
                            if cfg.SOLVER.FEATURE_INTERVAL == True:
                                
                                features_s = list(features_s)
                                features_s, features_t_aux = feature_interval(features_s, features_t_aux)

                            student_logit, loss_dict = predictor_s(images_strong, features_s, aux=None, targets=boxes)
                            ###################################################################

                            if not cfg.SOLVER.SFDA_REG_ON:
                                if "loss_reg" in loss_dict:
                                    loss_dict.pop("loss_reg")
                            if not cfg.SOLVER.SFDA_CTR_ON:
                                if "loss_centerness" in loss_dict:
                                    loss_dict.pop("loss_centerness")

                            loss_dict = {k + "_student": loss_dict[k] for k in loss_dict}
                            loss_dict_all = loss_dict

                            losses = sum(loss for loss in loss_dict_all.values())

                            # reduce losses over all GPUs for logging purposes
                            loss_dict_reduced = reduce_loss_dict(loss_dict_all)
                            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                            meters.update(loss=losses_reduced, **loss_dict_reduced)
                            
                            if update_part == "both":
                                optimizer["generator_s"].zero_grad()
                            optimizer["predictor_s"].zero_grad()
                            losses.backward()

                            if update_part == "both":
                                optimizer["generator_s"].step()
                            optimizer["predictor_s"].step()

                            if pytorch_1_1_0_or_later:
                                if update_part == "both":
                                    scheduler["generator_s"].step()
                                scheduler["predictor_s"].step()


            if iteration % cfg.SOLVER.SFDA_TEACHER_UPDATE_ITER == 0:
                update_teacher(cfg, generator_t, generator_s, predictor_t, predictor_s)
            
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr_G_s: {lr_G_s:.6f}",
                            # "lr_P_t_aux: {lr_P_t_aux:.6f}",
                            "lr_P_s: {lr_P_s:.6f}",
                            "max mem: {memory:.0f}",
                            "inconfident bank: {length}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_G_s=optimizer["generator_s"].param_groups[0]["lr"],
                        # lr_P_t_aux=optimizer["predictor_t_aux"].param_groups[0]["lr"],
                        lr_P_s=optimizer["predictor_s"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        length = len(inconfident_bank),
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if data_loaders_val is not None and test_period > 0 and iteration % test_period == 0:
                synchronize()
                for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
                    results, _ = inference(
                        model,
                        data_loader_val,
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                    )
                ap_tmp, ap50_tmp = results.results['bbox']['AP'], results.results['bbox']['AP50']
                # if ap50_tmp > ap50:
                #     ap50 = ap50_tmp
                #     checkpointer.save("model_best_ap50", **arguments)
                #     logger.info("Updating Best mAP50 Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                if ap_tmp > ap:
                    ap = ap_tmp
                    checkpointer.save("model_best_ap", **arguments)
                    logger.info("Updating Best mAP Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                synchronize()
                for k in model:
                    model[k].train()
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )

def get_ins_feats(cfg, features_t, boxes, idx):
    locations_per_im = boxes[idx].get_field("locations")
    levels_per_im = boxes[idx].get_field("levels")
    strides_per_im = torch.tensor([cfg.MODEL.FCOS.FPN_STRIDES[int(i)] for i in list(levels_per_im)]).float().to(locations_per_im.device)
    strides_per_im = strides_per_im.unsqueeze(-1)
    locations_per_im //= strides_per_im
    features_t_nograd = []
    for features_t_per_level in features_t:
        features_t_nograd.append(features_t_per_level[idx].clone().detach())
                        
                    # map locations to features
    ins_feats = []
    for idx_lvl, lvl in enumerate(levels_per_im):
        features_t_per_level = features_t_nograd[int(lvl)]
        x, y = int(locations_per_im[idx_lvl][0]), int(locations_per_im[idx_lvl][1])
        ins_feat = features_t_per_level[:, y, x]
        ins_feats.append(ins_feat)
    return idx,ins_feats

def add_to_ir(inconfident_bank, images_weak, images_strong, boxes, idx, dir=None):
    if dir:
        save_image(images_weak.tensors[idx].cpu(), dir)
    if len(inconfident_bank) == 20:
        del inconfident_bank[0]
    inconfident_bank.append(tuple([images_weak.tensors[idx].cpu(), images_strong.tensors[idx].cpu(), boxes[idx]]))

def update_teacher(cfg, generator_t, generator_s, predictor_t, predictor_s):
    if cfg.SOLVER.TEACHER_UPDATE_METHOD == 'decouple': # default: decouple
        generator_t, predictor_t = _decouple_teacher_model(generator_t, generator_s, predictor_t, predictor_s, keep_rate=cfg.SOLVER.SFDA_EMA_KEEP_RATE)
    elif cfg.SOLVER.TEACHER_UPDATE_METHOD == 'ema':
        generator_t, predictor_t = _update_teacher_model(generator_t, generator_s, predictor_t, predictor_s, keep_rate=cfg.SOLVER.SFDA_EMA_KEEP_RATE)

def feature_interval(features_s, features_t_aux):
    features_t_aux = list(features_t_aux)
    for i in range(len(features_s)):
        features_s[i] = features_s[i] * torch.sigmoid(features_t_aux[i].clone().detach())
    features_s = tuple(features_s)
    features_t_aux = tuple(features_t_aux)
    return features_s,features_t_aux
