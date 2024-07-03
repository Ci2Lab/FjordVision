import copy
import itertools
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            print(f"Prepared results for {iou_type}: {results[:5]}")  # Debug print
            
            coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            
            coco_eval.evaluate()
            eval_imgs = coco_eval.evalImgs

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            eval_imgs = self.eval_imgs[iou_type]
            eval_imgs = list(itertools.chain(*eval_imgs))
            self.eval_imgs[iou_type] = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown IoU type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]

            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        print(f"Prepared {len(coco_results)} results for COCO detection")  # Debug print
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            masks = prediction["masks"]
            for i in range(len(masks)):
                mask = masks[i]
                mask_cpu = mask[0, :, :, np.newaxis].cpu().numpy()
                rle = mask_util.encode(np.array(mask_cpu, dtype=np.uint8, order="F"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                
                # Debug: Check mask values
                print(f"Mask shape: {mask_cpu.shape}, unique values: {np.unique(mask_cpu)}")
                print(f"RLE: {rle}")

                coco_result = {
                    "image_id": original_id,
                    "category_id": prediction["labels"][i].item(),
                    "segmentation": rle,
                    "score": prediction["scores"][i].item(),
                }
                coco_results.append(coco_result)

        print(f"Prepared {len(coco_results)} results for COCO segmentation")  # Debug print
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]
            keypoints = prediction["keypoints"]

            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()
            keypoints = keypoints.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        print(f"Prepared {len(coco_results)} results for COCO keypoint")  # Debug print
        return coco_results

    def evaluate(self):
        all_img_ids = []
        all_eval_imgs = []
        for iou_type, coco_eval in self.coco_eval.items():
            coco_eval.accumulate()
            coco_eval.summarize()
            all_img_ids.extend(coco_eval.params.imgIds)
            all_eval_imgs.extend(coco_eval.evalImgs)

        return all_img_ids, all_eval_imgs
