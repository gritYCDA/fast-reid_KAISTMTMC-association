# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager

import torch

from fastreid.utils import comm
from fastreid.utils.logger import log_every_n_seconds

import re
from collections import defaultdict

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


# class DatasetEvaluators(DatasetEvaluator):
#     def __init__(self, evaluators):
#         assert len(evaluators)
#         super().__init__()
#         self._evaluators = evaluators
#
#     def reset(self):
#         for evaluator in self._evaluators:
#             evaluator.reset()
#
#     def process(self, input, output):
#         for evaluator in self._evaluators:
#             evaluator.process(input, output)
#
#     def evaluate(self):
#         results = OrderedDict()
#         for evaluator in self._evaluators:
#             result = evaluator.evaluate()
#             if is_main_process() and result is not None:
#                 for k, v in result.items():
#                     assert (
#                             k not in results
#                     ), "Different evaluators produce results with the same key {}".format(k)
#                     results[k] = v
#         return results


def inference_on_dataset(model, data_loader, evaluator, flip_test=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def pickle_reid(model, data_loader, flip_test=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    infos = []
    feats = []
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            infos.extend(inputs['img_paths'])
            feats.append(outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )
    feats = torch.cat(feats)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    
    pattern = re.compile(r's([-\d]+)_([\d]+)_c([\d]+)_([\d]+).jpg')
    sid_sets = set()
    feats_dict = defaultdict(list)
    for i, info in enumerate(infos):
        sid, fid, camid, pid = map(int, pattern.search(info).groups())
        sid_sets.add(sid)
        feats_dict[(sid, camid, pid)].append((fid, feats[i]))
        # feats_dict[(sid, camid, fid)].append((pid, feats[i]))
    sid_sets = list(sid_sets)
    feature_pkl_root = '/home/miruware/shw/prj-mtmc/work_dirs/clustering/config_runs'
    import os
    import os.path as osp
    import pickle
    for sid in sid_sets:
        outdir = osp.join(feature_pkl_root, f'kaist_mtmdc_qdtrack_s{sid}',
                          'pickled_appearance_features', 'test')
        if osp.exists(outdir):
            import shutil
            shutil.rmtree(outdir)
        os.makedirs(outdir)

    for i, (info, feats) in enumerate(feats_dict.items()):
        # print(i, len(feats_dict))
        sid, camid, pid = info
        feats = [_[1] for _ in feats]
        avg_feat = torch.stack(feats).mean(dim=0).cpu().numpy()

        outdir = osp.join(feature_pkl_root, f'kaist_mtmdc_qdtrack_s{sid}',
                          'pickled_appearance_features', 'test')
        filename = 'pid_{}_camid_{}.pkl'.format(pid, camid)
        feature_pickle_path = osp.join(outdir, filename)
        with open(feature_pickle_path, 'wb') as handle:
            pickle.dump(avg_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    
    # for i, (info, feats) in enumerate(feats_dict.items()):
    #     print(i, len(feats_dict))
    #     sid, camid, fid = info
    #     pid_to_feat = dict()
    #     import pdb
    #     pdb.set_trace()
    #     for feat in feats:
    #         pid = feat[0]
    #         feat = feat[1]
    #         pid_to_feat[pid] = feat
    #     outdir = osp.join(feature_pkl_root, f'kaist_mtmdc_qdtrack_s{sid}',
    #                       'pickled_appearance_features', 'test')
    #     filename = 'frameno_{}_camid_{}.pkl'.format(fid, camid)
        
    #     feature_pickle_path = osp.join(outdir, filename)
    #     with open(feature_pickle_path, 'wb') as handle:
    #         pickle.dump(pid_to_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    # if results is None:
    #     results = {}
    # return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
