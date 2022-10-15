#!/usr/bin/env python3
"""module class Yolo."""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """class Yolo."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor.

        Args:
            model_path (str): path to where a Darknet Keras model is stored
            classes_path (str): path to where the list of class names used for
                                the Darknet model, listed in order of index,
                                can be found
            class_t (float): represents the box score threshold for the
                             initial filtering step
            nms_t (float): representing the IOU threshold for non-max
                           suppression
            anchors (ndarray): contains all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process outputs method.

        Args:
            outputs (ndarray): contains the predictions from the Darknet model
                               for a single image
            image_size (ndarray): contains the image's original size
        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_h in range(pred.shape[0]):
                for grid_w in range(pred.shape[1]):
                    bx = ((self.sigmoid(pred[grid_h, grid_w, :, 0]) + grid_w)
                          / pred.shape[1])
                    by = ((self.sigmoid(pred[grid_h, grid_w, :, 1]) + grid_h)
                          / pred.shape[0])
                    anchor_tensor = self.anchors[ipred].astype(float)
                    anchor_tensor[:, 0] *= np.exp(pred[grid_h, grid_w, :, 2])\
                        / self.model.input.shape[1].value
                    anchor_tensor[:, 1] *= np.exp(pred[grid_h, grid_w, :, 3])\
                        / self.model.input.shape[2].value
                    pred[grid_h, grid_w, :, 0] = (bx - (anchor_tensor[:, 0] /
                                                  2)) * image_size[1]
                    pred[grid_h, grid_w, :, 1] = (by - (anchor_tensor[:, 1] /
                                                  2)) * image_size[0]
                    pred[grid_h, grid_w, :, 2] = (bx + (anchor_tensor[:, 0] /
                                                  2)) * image_size[1]
                    pred[grid_h, grid_w, :, 3] = (by + (anchor_tensor[:, 1] /
                                                  2)) * image_size[0]
        box_confidences = [self.sigmoid(pred[:, :, :, 4:5]) for pred in
                           outputs]
        box_class_probs = [self.sigmoid(pred[:, :, :, 5:]) for pred in outputs]
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes method.

        Args:
            boxes (ndarray): contains the processed boundary boxes for each
                             output
            box_confidences (ndarray): contains the processed box confidences
                                       for each output
            box_class_probs (ndarray): contains the processed box class
                                       probabilities for each output
        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_score = []
        bc = box_confidences
        bcp = box_class_probs
        for box_conf, box_probs in zip(bc, bcp):
            score = (box_conf * box_probs)
            box_score.append(score)
        box_classes = [s.argmax(axis=-1) for s in box_score]
        box_class_l = [b.reshape(-1) for b in box_classes]
        box_classes = np.concatenate(box_class_l)
        box_class_scores = [s.max(axis=-1) for s in box_score]
        b_scores_l = [b.reshape(-1) for b in box_class_scores]
        box_class_scores = np.concatenate(b_scores_l)
        mask = np.where(box_class_scores >= self.class_t)
        boxes_all = [b.reshape(-1, 4) for b in boxes]
        boxes_all = np.concatenate(boxes_all)
        scores = box_class_scores[mask]
        boxes = boxes_all[mask]
        classes = box_classes[mask]
        return (boxes, classes, scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non max suppression method.

        Args:
            filtered_boxes (ndarray): contains all of the filtered bounding
                                      boxes
            box_classes (ndarray): contains the class number for the class
                                   that filtered_boxes predicts
            box_scores (ndarray): contains the box scores for each box in
                                  filtered_boxes
        Returns:
            tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        f = []
        c = []
        s = []
        for i in (np.unique(box_classes)):
            idx = np.where(box_classes == i)
            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]
            keep = self.nms(filters, self.nms_t, scores)
            filters = filters[keep]
            scores = scores[keep]
            classes = classes[keep]
            f.append(filters)
            c.append(classes)
            s.append(scores)
        filtered_boxes = np.concatenate(f, axis=0)
        box_scores = np.concatenate(c, axis=0)
        box_classes = np.concatenate(s, axis=0)
        return (filtered_boxes, box_scores, box_classes)

    def nms(self, bc, thresh, scores):
        """
        Compute the index.

        Args:
            bc: box coordinates
            thresh: threshold
            scores: scores for each box indexed and sorted
        Returns:
            sorted index score for non max supression
        """
        x1 = bc[:, 0]
        y1 = bc[:, 1]
        x2 = bc[:, 2]
        y2 = bc[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while (order.size > 0):
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
