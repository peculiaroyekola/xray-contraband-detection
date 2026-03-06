"""
**EXERCISE**

This file contains the pipelines to training or test a Faster R-CNN model on RGB images. By default, the
orbs dataset is used for this pipeline which has been annotated by the LabelImg annotation software. The images are
resized to fit the model's input size. The main pipeline defines three different pipelines: training, validation
and testing. The main pipeline looks like this:

.. raw:: html
   :file: ../../diagrams/dl_object_detection_faster_rcnn/main.html

First the network is trained and each 10th epoch the results are validated on a separate set. The model with the
lowest loss on the validation set is saved to fasterrcnn.pth (early stopping). The training and validation loss are
shown in a tensorboard session. In our training session the results like this [#workdir]_.:

.. list-table::
   :align: center

   * - .. figure:: ../../../images/dl_object_detection_faster_rcnn/training_loss_fasterrcnn.png
          :scale: 70 %

          Training loss per epoch

     - .. figure:: ../../../images/dl_object_detection_faster_rcnn/valid_loss_fasterrcnn.png
          :scale: 70 %

          Validation loss per epoch

The testing pipeline loads the model and estimated bounding boxes on the full image. The testing pipeline outputs the
metrics Precision, Recall and F1-Score. The results are shows in a tensorboard session. In our run the results were
like this:

.. list-table::
   :align: center

   * - .. figure:: ../../../images/dl_object_detection_faster_rcnn/test_predictions.png
          :scale: 30 %

          Predictions from testing

     - .. figure:: ../../../images/dl_object_detection_faster_rcnn/test_targets.png
          :scale: 50 %

          Ground Truths

.. rubric:: Frequent Issues

**1: Faster R-CNN under the hood performs normalization on the image (again), see /home/YOUR_USERNAME/anaconda/envs/deeplearning/lib/python3.11/site-packages/torchvision/models/detection/transform.py line 129**
**2: Faster R-CNN always sees label 0 as background, thus your annotation labels must always start with 1. It will not give output boxes with label 0**

.. rubric:: Footnotes
.. [#workdir] All information regarding the runs is saved in the working directory (trained model and tensorboard)
"""


import os
from functools import partial
import numpy as np
import pandas as pd
import cv2

import torch
from torchvision.ops import nms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations import Resize, LongestMaxSize, PadIfNeeded


from common.data.datatypes import BoundingBox, SampleContainer
from common.data.datasets_info import supervised_object_detection_datasets, ABCDatasetInfo
from common.data.transforms import RemapLabels

from common.elements.utils import (
    get_tmp_dir,
    static_var,
    wait_forever,
    get_cicd_test_type,
    CICDTestType,
    reproduce_seed,
)

from elements.load_data import (
    get_dataloader_object_detection,
    convert_annotation_to_fasterrcnn_format,
)
from elements.run_training_xray import WORKING_DIR
from elements.save_model import save_model_pt
from elements.load_model import load_model_pt

from elements.optimize import (
    back_prop_pt,
    get_adam_optimizer_pt,
)

from elements.tune_params import (
    get_reduce_lr_on_plateau_pt,
    tune_learning_rate_pt,
)

from elements.predict import decode_rcnn_boxes_pt
from elements.save_model import save_model_pt

from elements.visualize import (
    create_tb,
    show_loss_tb,
    log_loss,
    show_image_and_boxes_tb,
    show_metrics_graph_tb,
    show_text_tb,
)

from elements.calc_metrics import (
    calc_mean_average_precision,
    calc_overall_precision_recall_f1score_prob_range,
    calc_per_class_precision_recall_f1score_prob_range,
)
from elements.preprocess import Log2_filter

def dummy_init_args(self):
    return []

for _cls in (
    A.OneOf, A.SomeOf, A.ReplayCompose, A.Compose, A.Sequential,
    A.GaussianBlur, A.MotionBlur, A.MedianBlur, A.Blur,
    A.GaussNoise, A.RandomBrightnessContrast, A.CLAHE,
    A.RandomGamma, A.ShiftScaleRotate, A.CoarseDropout,
    A.ImageCompression, A.HueSaturationValue,
):
    if not hasattr(_cls, "get_transform_init_args_names"):
        _cls.get_transform_init_args_names = dummy_init_args


# IMAGE PREPROCESSING

class GrayToRGB:
    def __call__(self, sample):
        img = sample.image_data.get()
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)
        sample.image_data = img
        return sample



class FilterEmptyAnnotations:
    def __call__(self, sample):
        boxes = sample.annotations.get(BoundingBox)
        if boxes is None or len(boxes) == 0:
            return None
        return sample

class ProperXrayNormalize:
    def __call__(self, sample):
        img = sample.image_data.get()

        if img.dtype == np.uint16 or img.max() > 300:
            img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        sample.image_data = img
        return sample

# Augmentation processsing
import albumentations as A
class ProperAugmentation:
 """
 Correctly applies Albumentations augmentations on normalized images:
 - Denormalizes to uint8
 - Applies augmentations (which expect uint8 or [0-1] float)
 - Renormalizes back to ImageNet stats
 """
 def __init__(self, aug_name: str):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Convert normalized float → uint8
        to_uint8 = A.Lambda(
            name="to_uint8",
            image=lambda img, **kwargs: np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8),
            p=1.0
        )
        # Convert uint8 back → normalized float
        back_to_norm = A.Lambda(
            name="back_to_norm",  # ← ADD THIS
            image=lambda img, **kwargs: ((img / 255.0) - mean) / std,
            p=1.0
        )

        transforms = []
        if aug_name == "GaussianBlur":
            transforms = [A.GaussianBlur(blur_limit=(7, 15), p=1.0)]
        elif aug_name == "CLAHE":
            transforms = [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)]
        elif aug_name == "Gamma":
            transforms = [A.RandomGamma(gamma_limit=(80, 120), p=1.0)]
        elif aug_name == "Emboss":
            transforms = [A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0)]
        elif aug_name == "NoAugment":
            transforms = []
        else:
            raise ValueError(f"Unknown augmentation: {aug_name}")

        geometric = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ),
        ]

        #  A.ReplayCompose to avoid serialization issues
        self.transform = A.Compose(
            [to_uint8] + geometric + transforms + [back_to_norm],
            p=1.0
            )


      def __call__(self, sample):
        img = sample.image_data.get()  # numpy array, normalized
        transformed = self.transform(image=img)
        sample.image_data = transformed["image"]
        return sample

# MODEL CREATION

def get_faster_rcnn_model(num_classes: int, device="cuda:0"):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

# UTILITIES

def _get_working_dir():
    return get_tmp_dir(os.path.splitext(os.path.basename(__file__))[0])

def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xg1, yg1, xg2, yg2 = box2

    xi1 = max(x1, xg1)
    yi1 = max(y1, yg1)
    xi2 = min(x2, xg2)
    yi2 = min(y2, yg2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1) * (y2 - y1) + (xg2 - xg1) * (yg2 - yg1) - inter + 1e-8
    return inter / union


# EARLY STOPPING implementation

class EarlyStopping:
    def __init__(self, patience=10, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        improved = value < self.best if self.mode == "min" else value > self.best

        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience




#Testing phase
def run_testing(test_ds_info: ABCDatasetInfo, preprocess_test: list, working_dir=_get_working_dir(),
                dev: str = "cuda:0", wait=False, aug_name: str = "default"):
    """
    This method implements the testing pipeline and spins up a tensorboard where the results are displayed.
    It shows the mean Average Precision for a single value of the iou (set to 0.5).

    .. raw:: html
        :file: ../../diagrams/dl_object_detection_faster_rcnn/test.html

    :param test_ds_info: DatasetInfo object containing the information (data directories, etc.) of the training data.
    :param working_dir: the current working directory where all information is stored (model weights, tensorboard, etc.)
    :param dev: cuda device
    :param preprocess_test: list of data preprocessing functions
    :param wait: wait at the end of the pipeline (True) or terminate program (False). Set wait to True when it is required to investigate the tensorboard after the pipeline finishes.
    """

    print("Working directory is:", working_dir)

    os.chdir(working_dir)

    # Load test dataloader
    test_dataloader = get_dataloader_object_detection(
        ds_info=test_ds_info, preprocessing=preprocess_test, batch_size=2, shuffle=False
    )
    # Get object class names
    object_class_names = test_dataloader.dataset.get_class_names()
    num_classes = len(object_class_names) + 1  # +1 for background → 34 total

    # For TensorBoard to show each classes as numbers
    class_numbers = [str(i) for i in range(num_classes)]  # ["0", "1", ..., "33"]

    # Create TensorBoard writer
    writer = create_tb("tensorboard_testing")

    # run testing load model
    object_class_names = test_dataloader.dataset.get_class_names()
    num_classes = len(object_class_names) + 1
    model = get_faster_rcnn_model(num_classes, dev)

    best_path = os.path.join(working_dir, "models", "fasterrcnn.pth")
    last_path = os.path.join(working_dir, "models", "fasterrcnn_last.pth")

#debug to ensuring the model is being saved and also to ensure early stopping actually worked
    if os.path.exists(best_path):
        print("Loading BEST model")
        load_model_pt(model, best_path)
    elif os.path.exists(last_path):
        print("Best model not found — loading LAST epoch model")
        load_model_pt(model, last_path)
    else:
        raise RuntimeError("No model checkpoint found for testing.")

    print(f"Model loaded: {num_classes} classes (including background)")
    model.eval()

    result_boxes_per_image = {}
    target_boxes_per_image = {}
    # This creates the filtered dict for the clean per class F1
    filtered_boxes_per_image = {}
    for i, batch in enumerate(test_dataloader):
        # Load data
        input_data = batch.image_data.get().to(dev)
        annotations = batch.annotations.get(BoundingBox).to(dev)
        input_data_np = input_data.cpu().numpy()

        # Convert annotations to Faster R-CNN format
        target_data = convert_annotation_to_fasterrcnn_format(annotations=annotations, dev=dev)

        # prediction with post processing
        with torch.no_grad():
            output_data = model(input_data)
            raw_result_boxes = []
            filtered_result_boxes = []

        for img_idx, pred in enumerate(output_data):
            scores = pred['scores']
            labels = pred['labels']
            boxes = pred['boxes']

            # Remove background
            keep = labels > 0
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if len(scores) == 0:
                empty = np.empty((0, 6))
                raw_result_boxes.append(empty)
                filtered_result_boxes.append(empty)
                continue

            # Apply NMS on ALL predictions
            keep = nms(boxes, scores, iou_threshold=0.5)
            boxes = boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            labels = labels[keep].cpu().numpy()

            all_boxes = np.column_stack([
                boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2], labels, scores
            ])

            raw_result_boxes.append(all_boxes)
            filtered_result_boxes.append(all_boxes[scores >= 0.5])

        # Decode ground truth
        target_boxes = decode_rcnn_boxes_pt(target_data)

        # PROCESS EACH IMAGE IN BATCH
        for batch_index, (input_data_item, raw_boxes_item, filtered_boxes_item, target_boxes_item) in enumerate(
                zip(input_data, raw_result_boxes, filtered_result_boxes, target_boxes)):
            idx = (i * test_dataloader.batch_size) + batch_index
            in_filename = test_dataloader.dataset[idx].image_fpath

            # VISUALIZATION
            show_image_and_boxes_tb(
                input_data_item.cpu().numpy(),
                image_name=os.path.basename(in_filename),
                boxes_result=filtered_boxes_item,
                boxes_target=target_boxes_item,
                epoch=idx,
                name="result and target boxes",
                writer=writer,
                dataformats="CHW",
                normalize=True,
                class_names=class_numbers
            )

            # Save for mAP for ALL predictions
            result_boxes_per_image[in_filename] = raw_boxes_item

            # Save for clean per-class F1
            filtered_boxes_per_image[in_filename] = filtered_boxes_item

            # Also save target
            target_boxes_per_image[in_filename] = target_boxes_item


   #To show the summary before the matric is being shown. this explain what the matric shown is for
    print("\n" + "=" * 60)
    print("Testing Results Summary")
    print("=" * 60)
    total_preds = sum(len(boxes) for boxes in result_boxes_per_image.values())
    total_gts = sum(len(boxes) for boxes in target_boxes_per_image.values())
    print(f"Total predictions: {total_preds}")
    print(f"Total ground truth: {total_gts}")

    if total_preds > 0:
        pred_labels = []
        for filename, boxes in result_boxes_per_image.items():
            for p in boxes:
                if len(p) >= 6:
                    label = int(p[4])
                    pred_labels.append(label)
        print(f"Predicted class labels: {sorted(set(pred_labels))}")

    if total_gts > 0:
        gt_labels = []
        for boxes in target_boxes_per_image.values():
            for box in boxes:
                if len(box) >= 5:
                    gt_labels.append(int(box[4]))
        print(f"Ground truth class labels: {sorted(set(gt_labels))}")
    print("=" * 60 + "\n")


    # METRICS COMPUTATION

    # 1. Overall PR curve over thresholds → use ALL predictions
    metrics = calc_overall_precision_recall_f1score_prob_range(
        result_boxes_per_image=result_boxes_per_image,  # ← ALL
        target_boxes_per_image=target_boxes_per_image,
        iou_t=0.5,
        num_classes=num_classes
    )
    show_metrics_graph_tb(metrics=metrics, writer=writer)
    # CORRECT PER-CLASS METRICS + DETAILED TABLE
    try:
        per_class_ap = {}
        best_f1_per_class = {}
        best_thr_per_class = {}
        best_recall_per_class = {}
        metrics_at_05 = {}

        class_names = ["background"] + test_dataloader.dataset.get_class_names()

        for c in range(1, num_classes):
            preds = []
            gts = []
            for filename in result_boxes_per_image:
                for box in result_boxes_per_image[filename]:
                    if len(box) < 6: continue
                    y1, x1, y2, x2, label, score = map(float, box[:6])
                    if int(label) == c:
                        preds.append((score, [x1, y1, x2, y2]))  # (score, box)
                for box in target_boxes_per_image[filename]:
                    if len(box) < 5: continue
                    y1, x1, y2, x2, label = map(float, box[:5])
                    if int(label) == c:
                        gts.append([x1, y1, x2, y2])

            if len(gts) == 0 or len(preds) == 0:
                per_class_ap[c] = 0.0
                best_f1_per_class[c] = 0.0
                best_thr_per_class[c] = 0.00
                metrics_at_05[c] = {"recall": 0.0, "precision": 0.0, "tp": 0, "fp": 0, "fn": len(gts), "fptp": 0.0}
                continue

            # FULL PR CURVE FOR BEST F1, THR, AP all at 0.5
            preds.sort(reverse=True)  # By score descending
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            matched = set()

            for i, (score, pbox) in enumerate(preds):
                best_iou = 0
                best_j = -1
                for j, gbox in enumerate(gts):
                    if j in matched: continue
                    iou_val = bbox_iou(pbox, gbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j = j
                if best_iou >= 0.5 and best_j != -1:
                    tp[i] = 1
                    matched.add(best_j)
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / len(gts)
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

            best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
            best_f1 = f1_scores[best_idx] if len(f1_scores) > 0 else 0.0
            best_thr = preds[best_idx][0] if len(preds) > 0 else 0.0
            best_rec = recall[best_idx] if len(recall) > 0 else 0.0

            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11.0

            # METRICS @ THR=0.5
            filtered_preds = [p for p in preds if p[0] >= 0.5]
            if len(filtered_preds) == 0:
                tp5 = 0
                fp5 = 0
            else:
                filtered_preds.sort(reverse=True)
                tp5_arr = np.zeros(len(filtered_preds))
                fp5_arr = np.zeros(len(filtered_preds))
                matched5 = set()
                for i, (score, pbox) in enumerate(filtered_preds):
                    best_iou = 0
                    best_j = -1
                    for j, gbox in enumerate(gts):
                        if j in matched5: continue
                        iou_val = bbox_iou(pbox, gbox)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_j = j
                    if best_iou >= 0.5 and best_j != -1:
                        tp5_arr[i] = 1
                        matched5.add(best_j)
                    else:
                        fp5_arr[i] = 1
                tp5 = np.sum(tp5_arr)
                fp5 = np.sum(fp5_arr)

            fn5 = len(gts) - tp5
            prec5 = tp5 / (tp5 + fp5 + 1e-8) if (tp5 + fp5) > 0 else 0.0
            rec5 = tp5 / len(gts) if len(gts) > 0 else 0.0
            fptp5 = fp5 / (tp5 + 1e-8)

            # to Save the results
            per_class_ap[c] = ap
            best_f1_per_class[c] = best_f1
            best_thr_per_class[c] = best_thr
            best_recall_per_class[c] = best_rec
            metrics_at_05[c] = {"recall": rec5, "precision": prec5, "tp": tp5, "fp": fp5, "fn": fn5, "fptp": fptp5}

        rows = []
        for c in range(1, num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            rows.append({
                "Class": name,
                "Best F1": round(best_f1_per_class.get(c, 0.0), 3),
                "Best F1 Thr": round(best_thr_per_class.get(c, 0.0), 2),
                "Recall@0.5": round(metrics_at_05.get(c, {"recall": 0.0})["recall"], 3),
                "Precision@0.5": round(metrics_at_05.get(c, {"precision": 0.0})["precision"], 3),
                "TP@0.5": int(metrics_at_05.get(c, {"tp": 0})["tp"]),
                "FP@0.5": int(metrics_at_05.get(c, {"fp": 0})["fp"]),
                "FN@0.5": int(metrics_at_05.get(c, {"fn": 0})["fn"]),
                "FP/TP@0.5": round(metrics_at_05.get(c, {"fptp": 0.0})["fptp"], 3)
            })

        df = pd.DataFrame(rows)
        table_md = df.to_markdown(index=False)
        show_text_tb(name="Per-Class Metrics", text=table_md, writer=writer)
        print("\n" + table_md)
        return rows

    except Exception as e:
        print(f"Per-class table skipped due to: {e}")
        show_text_tb(name="Per-Class Metrics", text="Table generation failed (likely empty classes)", writer=writer)
        return []



# validation stage
def run_validation(valid_loader, model, epoch, working_dir=_get_working_dir(), dev: str = "cuda:0", writer=None):
    print("Working directory is:", working_dir)

    os.chdir(working_dir)
    loss_sum, loss_cnt = 0, 0

    object_class_names = valid_loader.dataset.get_class_names()
    num_classes = len(object_class_names) + 1
    class_numbers = [str(i) for i in range(num_classes)]
    model.eval()

    for batch in valid_loader:
        input_data = batch.image_data.get().to(dev)
        annotations = batch.annotations.get(BoundingBox).to(dev)
        target_data = convert_annotation_to_fasterrcnn_format(annotations=annotations, dev=dev)
        input_data_np = input_data.cpu().numpy()

 # Compute validation loss
        model.train()
        with torch.no_grad():
            outputs = model(input_data, target_data)
            if isinstance(outputs, dict):
                loss = sum(loss for loss in outputs.values())
            else:
                loss = torch.tensor(0.0, device=dev)
        model.eval()

        # PREDICTION FOR VISUALIZATION
        if loss_cnt == 0:
            with torch.no_grad():
                output_data = model(input_data)

            # Use reasonable threshold for visualization
            result_boxes = decode_rcnn_boxes_pt(output_data, min_prob=0.5)
            target_boxes = decode_rcnn_boxes_pt(target_data)

            # Debug to print number of predicted boxes
            print(f"[VALID] Predicted boxes: {len(result_boxes[0])}, GT: {len(target_boxes[0])}")

            show_image_and_boxes_tb(
                input_data[0].cpu().numpy(),
                result_boxes[0],
                boxes_target=target_boxes[0],
                epoch=epoch,
                normalize=True,
                dataformats="CHW",
                writer=writer,
                class_names=class_numbers,
            )

        loss_sum += loss.cpu().item()
        loss_cnt += 1

    avg_loss = loss_sum / loss_cnt
    show_loss_tb(avg_loss, epoch, name="valid_loss", writer=writer)
    log_loss(avg_loss, epoch, name="valid_loss")

    if avg_loss < run_validation.best_loss:
        filename = os.path.abspath(os.path.join(working_dir, "models", "fasterrcnn.pth"))
        print(f"Saving best model to {filename}")
        save_model_pt(model, filename, loss=avg_loss, epoch=epoch)
        run_validation.best_loss = avg_loss

    return avg_loss

# training stage
def run_training(train_ds_info: ABCDatasetInfo, valid_ds_info: ABCDatasetInfo,
                 preprocess_train: list, preprocess_val: list,
                 num_epochs: int = 50, working_dir=_get_working_dir(), dev: str = "cuda:0"):
    """
    Train Faster R-CNN on the training set. Tensorboard will automatically be started to display training progress.
    Training is executed for 50 epochs.

    .. raw:: html
       :file: ../../diagrams/dl_object_detection_faster_rcnn/train.html

    :param train_ds_info: DatasetInfo object containing the information (data directories, etc.) of the training data.
    :param valid_ds_info: DatasetInfo object containing the information (data directories, etc.) of the validation data.
    :param preprocess_train_val: list of data preprocessing functions
    :param num_epochs: the amount of epochs the model should train for
    :param working_dir: the current working directory where all information is stored (model weights, tensorboard, etc.)
    :param dev: the cuda device to training on
    """

    print("Working directory is:", working_dir)
    os.chdir(working_dir)

    # Dataloaders
    train_dataloader = get_dataloader_object_detection(
        ds_info=train_ds_info,
        preprocessing=preprocess_train,  # ← with augmentations
        batch_size=2,
        shuffle=True
    )
    valid_dataloader = get_dataloader_object_detection(
        ds_info=valid_ds_info,
        preprocessing=preprocess_val,  # ← clean, no random aug
        batch_size=2,
        shuffle=False
    )

    # MODEL CREATION FOR TRAINING
    object_class_names = train_dataloader.dataset.get_class_names()
    num_classes = len(object_class_names) + 1  # +1 background

    model = get_faster_rcnn_model(num_classes, dev)


    if not hasattr(run_validation, "best_loss"):
        run_validation.best_loss = float("inf")
        print("run_validation.best_loss initialized")


    os.makedirs("models", exist_ok=True)
    writer = create_tb("tensorboard")


    # Optimizer & scheduler
    optimizer = get_adam_optimizer_pt(model, learnrate=0.0001)
    scheduler = get_reduce_lr_on_plateau_pt(optimizer, patience=20)

    early_stopper = EarlyStopping(patience=5, mode="min")

    for epoch in range(num_epochs):
        model.train()
        loss_sum, loss_cnt = 0, 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_data = batch.image_data.get().to(dev)
            annotations = batch.annotations.get(BoundingBox).to(dev)
            target_data = convert_annotation_to_fasterrcnn_format(annotations=annotations, dev=dev)

            loss_dict = model(input_data, target_data)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_cnt += 1

        avg_loss = loss_sum / loss_cnt
        show_loss_tb(avg_loss, epoch, writer=writer)
        log_loss(avg_loss, epoch)
        tune_learning_rate_pt(avg_loss, scheduler)

        if epoch >= 10:  # Validate every epoch starting from epoch 10 (warm-up phase)
            model.eval()
            val_loss = run_validation(
                valid_dataloader, model, epoch, working_dir, dev, writer=writer
            )
            model.train()

            # EARLY STOPPING CHECK
            if early_stopper.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    last_model_path = os.path.join(working_dir, "models", "fasterrcnn_last.pth")
    torch.save(model.state_dict(), last_model_path)

def produce_results(img_dir=supervised_object_detection_datasets.orbs_training.images_dir) -> str:
    """
    This functions receives a file containing location of a dataset. This dataset is in the LabelImg format,which means that it can be loaded with the :meth:~elements.load_data.basic.get_labelimg_dataloader_pt. You
    should process all images in the list using your :meth:dl_object_detection_faster_rcnn.run_testing pipeline and
    then this function should return a textfile that stores the bounding boxes that were found in each image.

    :param img_dir: folder containing the testing images. :return: Output filename with one line per image. The line
        should start with the absolute filename and then the bounding boxes. For each bounding box the y1, x1, y2, x2,class_id and class_prob should be written. This output file can be generated with

    :meth:~elements.save_results.basic.save_boxes_to_textfile. ::
            absolute_filename y1 x1 y2 x2 class_id class_prob y1 x1 y2 x2 class_id class_prob ... \newline
            absolute_filename y1 x1 y2 x2 class_id class_prob y1 x1 y2 x2 class_id class_prob ... \newline
            ...

        Note: the filenames should contain absolute paths for the grading system to be able to find them.

    :example:

    >>> import os
    >>> os.path.isfile(produce_results())
    True
    """
    return os.path.join(_get_working_dir(), "output_test", "boxes_per_image.txt")


def _test(test_only: bool = False, wait=False):
    """
    This method runs the training, validation, testing pipeline on the Orbs dataset.
    """

    torch.cuda.empty_cache()
    reproduce_seed(42)
# for testing with no augmentation in them.
    preprocess = [
            RemapLabels(
            mapping={"surgical_implant": 1,
                     "shoe": 2,
                     "dental": 3,
                     "zipper": 4,
                     "zipper_buttons": 5,
                     "bag": 6,
                     "lighter": 7,
                     "cast": 8,
                     "glasses": 9,
                     "bra": 10,
                     "charger": 11,
                     "watch": 12,
                     "drugs": 13,
                     "ankle_monitor": 14,
                     "padlock": 15,
                     "paper": 16,
                     "key": 17,
                     "cup": 18,
                     "piercing": 19,
                     "belt": 20,
                     "jewelry": 21,
                     "cellphone": 22,
                     "cuffs": 23,
                     "unknown_anomaly": 24,
                     "prosthetic": 25,
                     "screw_pocket": 26,
                     "collar": 27,
                     "buckle": 28,
                     "crackpipe": 29,
                     "motion_artifact": 30,
                     "radiopaque": 31,
                     "abnormal_pose": 32,
                     "unknown_definedbyoperator": 33,
                   }
            , from_class_names=True),
       # Log2_filter,
        GrayToRGB(),
        ProperXrayNormalize(),
        LongestMaxSize(max_size=1333, p=1),
        PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    ]

    if get_cicd_test_type() == CICDTestType.FULL_COMPLETE:
        train_fn = run_training
    else:
        train_fn = partial(run_training, num_epochs=30)

    train_ds_info = supervised_object_detection_datasets.surgical_implants_training
    train_ds_info.mean, train_ds_info.std = [0, 0, 0], [1, 1, 1]
    print("Dataset class names:", train_ds_info.class_names)

    valid_ds_info = supervised_object_detection_datasets.surgical_implants_valid
    valid_ds_info.mean, valid_ds_info.std = [0, 0, 0], [1, 1, 1]
    if not test_only:
        train_fn(train_ds_info=train_ds_info, valid_ds_info=valid_ds_info, preprocess_train_val=preprocess)

    test_ds_info = supervised_object_detection_datasets.surgical_implants_test
    test_ds_info.mean, test_ds_info.std = [0, 0, 0], [1, 1, 1]
    run_testing(test_ds_info, preprocess, wait=wait)

if __name__ == "__main__":
    import pandas as pd

    torch.cuda.empty_cache()
    reproduce_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Base preprocessing (shared by all experiments) training
    base_preprocess = [
        FilterEmptyAnnotations(),
        RemapLabels(
            mapping={"surgical_implant": 1,
                     "shoe": 2,
                     "dental": 3,
                     "zipper": 4,
                     "zipper_buttons": 5,
                     "bag": 6,
                     "lighter": 7,
                     "cast": 8,
                     "glasses": 9,
                     "bra": 10,
                     "charger": 11,
                     "watch": 12,
                     "drugs": 13,
                     "ankle_monitor": 14,
                     "padlock": 15,
                     "paper": 16,
                     "key": 17,
                     "cup": 18,
                     "piercing": 19,
                     "belt": 20,
                     "jewelry": 21,
                     "cellphone": 22,
                     "cuffs": 23,
                     "unknown_anomaly": 24,
                     "prosthetic": 25,
                     "screw_pocket": 26,
                     "collar": 27,
                     "buckle": 28,
                     "crackpipe": 29,
                     "motion_artifact": 30,
                     "radiopaque": 31,
                     "abnormal_pose": 32,
                     "unknown_definedbyoperator": 33,
},
            from_class_names=True),
        #Log2_filter,
        GrayToRGB(),
        ProperXrayNormalize(),
        LongestMaxSize(max_size=1333, p=1),
        PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    ]
    train_ds_info = supervised_object_detection_datasets.surgical_implants_training
    valid_ds_info = supervised_object_detection_datasets.surgical_implants_valid
    test_ds_info  = supervised_object_detection_datasets.surgical_implants_test

    for ds in [train_ds_info, valid_ds_info, test_ds_info]:
        ds.mean, ds.std = [0, 0, 0], [1, 1, 1]

    experiments = [
        ("NoAugment", "NoAugment"),
        ("CLAHE", "CLAHE"),
        ("Gamma", "Gamma"),
        ("Emboss", "Emboss"),
        ("GaussianBlur", "GaussianBlur"),
    ]

    all_rows = []
    for key, display in experiments:
        print("\n" + "="*80)
        print(f"EXPERIMENT: {display}")
        print("="*80)

        # Fresh model and optimizer for EVERY experiment
        model = get_faster_rcnn_model(len(train_ds_info.class_names) + 1, device)
        optimizer = get_adam_optimizer_pt(model, learnrate=0.0005)
        scheduler = get_reduce_lr_on_plateau_pt(optimizer, patience=10)

        work_dir = f"/home/student/Faster_rcnn/PROPEREXPERIMENT/exp_{key}"
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, "models"), exist_ok=True)

        # Create proper augmentation ONLY for training
        if display == "NoAugment":
            aug = None
        else:
            aug = ProperAugmentation(display)

        # TRAINING preprocessing: includes the different augmentations
        preprocess_train = base_preprocess.copy()
        if aug is not None:
            preprocess_train.insert(4, aug)  # insert after Log2_filter, before normalize

        # VALIDATION & TEST preprocessing: NO augmentations
        preprocess_val_test = base_preprocess.copy()  # no aug inserted

        # Train from scratch
        run_training(
            train_ds_info=train_ds_info,
            valid_ds_info=valid_ds_info,
            preprocess_train=preprocess_train,  # ← with aug
            preprocess_val=preprocess_val_test,  # ← NEW: clean for val
            num_epochs=70,
            working_dir=work_dir
        )

        # Test with preprocessing
        rows = run_testing(
            test_ds_info=test_ds_info,
            preprocess_test=preprocess_val_test,  # ← clean, no random aug
            working_dir=work_dir,
            aug_name=display,
            wait=False
        )
        for r in rows:
            r["Augmentation"] = display
        all_rows.extend(rows)

        #  Debug to print ONLY the current experiment's table (not cumulative)
        if rows:
            current_df = pd.DataFrame(rows)
            cols = ["Augmentation", "Class", "Best F1", "Best F1 Thr", "Recall@0.5",
                    "Precision@0.5", "TP@0.5", "FP@0.5", "FN@0.5", "FP/TP@0.5"]
            current_df = current_df[cols]

            print("\n" + "=" * 80)
            print(f"TABLE AFTER {display.upper()} EXPERIMENT")
            print("=" * 80)
            print(current_df.to_string(index=False))
            print("\n" + "-" * 80)

    # Final table after the whole experiment has been done
    if all_rows:
        df = pd.DataFrame(all_rows)
        cols = ["Augmentation", "Class", "Best F1", "Best F1 Thr", "Recall@0.5",
                "Precision@0.5", "TP@0.5", "FP@0.5", "FN@0.5", "FP/TP@0.5"]
        df = df[cols]

        output_path = "/home/student/Faster_rcnn/PROPEREXPERIMENT/results_table.csv"
        df.to_csv(output_path, index=False)

        print("\n" + "="*100)
        print("EXPERIMENT FINISHED — FINAL TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print(f"\nSaved to: {output_path}")
    else:
        print("No results collected.")