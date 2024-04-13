import os 
import numpy as np
import torch 
import json
import random

from pathlib import Path
from torch_geometric.data import Data as pygData
from utils.oop import SingletonMeta
from .quickdraw_dataset import get_graph_data

from typing import List, Dict, Set, Tuple, Union, Optional, Any


class DoodleDatasetMeta(metaclass=SingletonMeta):
    def __init__(self):
        self.IND2CLS = ["car", "cloud", "fish", "flower", "sailboat", "sun", "tree", "<empty>"]
        self.CLS2IND = {cls: ind for ind, cls in enumerate(self.IND2CLS)}

        self.IND2SEGMENTLABEL = [
            'boat hull',
            'body',
            'center',
            'cloud',
            'crown',
            'eye',
            'fin',
            'hull',
            'leaf',
            'leave',
            'mast',
            'mouth',
            'other',
            'petal',
            'ray',
            'sail',
            'stem',
            'tail',
            'trunk',
            'wheel',
            "<empty>"
        ]
        self.SEGMENTLABEL2IND = {label: ind for ind, label in enumerate(self.IND2SEGMENTLABEL)}

        self.MAX_STROKE_NUM = 43
        self.MAX_POINTS_NUM = 256


class DoodleDataset(torch.utils.data.Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}
    meta = DoodleDatasetMeta()

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.all_paths = list(self.data_dir.glob(f"*/*.json"))

    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        path: Path = self.all_paths[idx]
        data = self._load_data(path)
        scaled_data = self._scale_strokes(data)

        key_id = torch.tensor(idx)
        category = torch.tensor(self.meta.CLS2IND[path.parent.name])
        seg_label = self._get_seg_label(scaled_data)
        seg_label1 = self._get_seg_label1(scaled_data)
        components_num = torch.unique(seg_label1).shape[0] - 1 # - 1 for <empty>
        stroke_mask = self._get_stroke_mask(scaled_data)
        position_list = self._get_position_list(scaled_data)
        points_offset = self._get_points_offsets(scaled_data)
        sketch_stroke_num = torch.tensor(len(data["lines"])) # Number of strokes
        stroke_number = self._get_stroke_num(scaled_data) # Vector with strokes order
        drawing = self._get_drawings(scaled_data)
        
        edge_index = get_graph_data(sketch_stroke_num)
        graph_data = pygData(
            x=torch.zeros((sketch_stroke_num, 2)),
            edge_index=torch.LongTensor(edge_index)
        )

        # One hot vector with 1 at the index of used segment labels
        seg_label2=torch.zeros((87)) # whtf is 87 ??? Max number of segment classes?
        for i in range(components_num):
            seg_label2[seg_label1[i]] = 1


        sample = {
            'points_offset': points_offset, 
            'category': category, 
            'seg_label': seg_label, 
            'seg_label1': seg_label1,
            'position_list': position_list,
            'stroke_number': stroke_number, 
            'stroke_mask': stroke_mask,
            'sketch_stroke_num': sketch_stroke_num,
            'graph_data': graph_data,
            'key_id': key_id,
            'seg_label2': seg_label2,
            
            # DEBUG 
            # Delete these lines after debugging
            "drawing": drawing,
            "strokes": scaled_data["lines"],
        }

        return sample

    def _load_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def _get_seg_label(self, data):
        int_labels = []
        for segment in data["segments"]:
            labels = set(segment)
            
            if len(labels) > 1:
                Warning(f"Multiple labels in segment: {labels}")
                
            int_labels.append(self.meta.SEGMENTLABEL2IND[labels.pop()])

        result = torch.full((self.meta.MAX_STROKE_NUM, ), fill_value=self.meta.SEGMENTLABEL2IND["<empty>"])
        result[:len(int_labels)] = torch.tensor(int_labels)
        return result
    
    def _get_seg_label1(self, data):
        classes: Set[str] = set()
        for segment in data["segments"]:
            # segment: List[str]
            classes.update(segment)

        ind_classes = [self.meta.SEGMENTLABEL2IND[cls] for cls in classes]
        result = torch.full((self.meta.MAX_STROKE_NUM, ), fill_value=self.meta.SEGMENTLABEL2IND["<empty>"])
        result[:len(ind_classes)] = torch.tensor(ind_classes)
        return result
    
    def _get_stroke_mask(self, data):
        num_strokes = len(data["lines"])
        result = torch.full((self.meta.MAX_STROKE_NUM + 1, ), fill_value=1.0)
        result[:num_strokes+1] = 0.0
        return result
    
    def _scale_strokes(self, data):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        data = data.copy()

        for line in data["lines"]:
            for x, y in line:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
        
        width = x_max - x_min
        height = y_max - y_min

        scaling_factor = max(width, height)

        for line in data["lines"]:
            for i, (x, y) in enumerate(line):
                x = (x - x_min) / scaling_factor
                y = (y - y_min) / scaling_factor
                line[i] = (x, y)

        return data
    

    def _get_position_list(self, data):
        result = torch.zeros(self.meta.MAX_STROKE_NUM, 2)
        
        for i, line in enumerate(data["lines"]):
            x, y = line[0]
            result[i, :2] = torch.tensor([x, y])
        
        return result

    def _get_points_offsets(self, data):
        result = torch.zeros(self.meta.MAX_STROKE_NUM, self.meta.MAX_POINTS_NUM, 4)

        for i, line in enumerate(data["lines"]):
            result[i, 0, 2] = 1.0 # set (0, 0, 1, 0) | (x, y, pen_down, pen_up)
            
            for j in range(1, len(line)):
                x_curr, y_curr = line[j]
                x_prev, y_prev = line[j-1]

                result[i, j, 0] = x_curr - x_prev # x
                result[i, j, 1] = y_curr - y_prev # y
                result[i, j, 2] = 1 # pen_down
                result[i, j, 3] = 0 # pen_up
            
            result[i, len(line), 3] = 1.0 # pen_up (0, 0, 0, 1.0) | (x, y, pen_down, pen_up)

        return result
    
    def _get_stroke_num(self, data):
        result = torch.zeros(self.meta.MAX_STROKE_NUM)
        strokes_num = len(data["lines"])
        strokes_num = min(strokes_num, self.meta.MAX_STROKE_NUM)

        result[:strokes_num] = torch.arange(start=1, end=strokes_num+1, step=1)
        return result
    
    def _get_drawings(self, data):
        result = torch.zeros(self.meta.MAX_POINTS_NUM, 3)
        ptr = 0
        for line in data["lines"]:
            if ptr + len(line) > self.meta.MAX_POINTS_NUM:
                break

            line = torch.tensor(line)
            result[ptr:ptr+len(line), :2] = line
            ptr += len(line)
            result[ptr-1, 2] = 1.0
        return result