"""
Multi-Object Tracker (MOT) for Underwater Video Feeds.
Assigns unique persistent IDs across consecutive frames and prevents double-counting.
"""

import numpy as np
from typing import List, Dict, Any

class UnderwaterTracker:
    def __init__(self, iou_threshold: float = 0.35, max_disappeared: int = 15):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.next_id = 1
        self.tracked_objects = {}
        self.cumulative_counts = {
            'marine_life': 0,
            'debris': 0,
            'infrastructure': 0,
            'diver': 0,
            'other': 0
        }
        self.seen_unique_labels = {}

    def _calculate_iou(self, boxA: list, boxB: list) -> float:
        yminA, xminA, ymaxA, xmaxA = boxA
        yminB, xminB, ymaxB, xmaxB = boxB

        inter_ymin = max(yminA, yminB)
        inter_xmin = max(xminA, xminB)
        inter_ymax = min(ymaxA, ymaxB)
        inter_xmax = min(xmaxA, xmaxB)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        areaA = max(0, xmaxA - xminA) * max(0, ymaxA - yminA)
        areaB = max(0, xmaxB - xminB) * max(0, ymaxB - yminB)
        union_area = areaA + areaB - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def update(self, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.tracked_objects:
            for obj in detected_objects:
                obj_id = self.next_id
                self.next_id += 1
                self.tracked_objects[obj_id] = {
                    'box': obj.get('box_2d', [0,0,0,0]),
                    'label': obj.get('label', 'Unknown'),
                    'category': obj.get('category', 'other'),
                    'disappeared': 0
                }
                obj['track_id'] = obj_id
                cat = obj.get('category', 'other')
                self.cumulative_counts[cat] = self.cumulative_counts.get(cat, 0) + 1
                lbl = obj.get('label', 'Unknown')
                self.seen_unique_labels[lbl] = self.seen_unique_labels.get(lbl, 0) + 1
            return detected_objects

        if not detected_objects:
            to_del = []
            for obj_id in self.tracked_objects:
                self.tracked_objects[obj_id]['disappeared'] += 1
                if self.tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                    to_del.append(obj_id)
            for d in to_del:
                del self.tracked_objects[d]
            return []

        tracked_ids = list(self.tracked_objects.keys())
        iou_matrix = np.zeros((len(tracked_ids), len(detected_objects)))

        for i, tid in enumerate(tracked_ids):
            for j, det in enumerate(detected_objects):
                iou_matrix[i, j] = self._calculate_iou(self.tracked_objects[tid]['box'], det.get('box_2d', [0,0,0,0]))

        matched_tracks = set()
        matched_dets = set()

        if len(tracked_ids) > 0 and len(detected_objects) > 0:
            indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
            for r, c in zip(indices[0], indices[1]):
                if r in matched_tracks or c in matched_dets:
                    continue
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_tracks.add(r)
                    matched_dets.add(c)
                    tid = tracked_ids[r]
                    self.tracked_objects[tid]['box'] = detected_objects[c].get('box_2d', [0,0,0,0])
                    self.tracked_objects[tid]['disappeared'] = 0
                    detected_objects[c]['track_id'] = tid

        for j, det in enumerate(detected_objects):
            if j not in matched_dets:
                obj_id = self.next_id
                self.next_id += 1
                self.tracked_objects[obj_id] = {
                    'box': det.get('box_2d', [0,0,0,0]),
                    'label': det.get('label', 'Unknown'),
                    'category': det.get('category', 'other'),
                    'disappeared': 0
                }
                det['track_id'] = obj_id
                cat = det.get('category', 'other')
                self.cumulative_counts[cat] = self.cumulative_counts.get(cat, 0) + 1
                lbl = det.get('label', 'Unknown')
                self.seen_unique_labels[lbl] = self.seen_unique_labels.get(lbl, 0) + 1

        for i, tid in enumerate(tracked_ids):
            if i not in matched_tracks:
                self.tracked_objects[tid]['disappeared'] += 1

        to_del = [tid for tid, data in self.tracked_objects.items() if data['disappeared'] > self.max_disappeared]
        for d in to_del:
            del self.tracked_objects[d]

        return detected_objects

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_unique_objects': self.next_id - 1,
            'currently_tracked': len(self.tracked_objects),
            'cumulative_by_category': self.cumulative_counts,
            'unique_species_count': len([k for k, v in self.tracked_objects.items() if v.get('category') == 'marine_life']),
            'unique_debris_count': len([k for k, v in self.tracked_objects.items() if v.get('category') == 'debris'])
        }
