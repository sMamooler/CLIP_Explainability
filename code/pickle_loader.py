"""
taken from Robin
"""

import os
import pickle
import random


class Pickle_data_loader :
    
    def __init__(self, path, num_samples, transformations = None ):
        self.path = path
        self.transformations = transformations if transformations is not None else []
        self.num_samples = num_samples
        
    def load_label(self, label, batch_size = 40):
        batch_id, label = divmod(label, batch_size)
        batch = self._open_pickle(batch_id)
        return {key : values[label] for key, values in batch.items()}
        
    def __iter__(self):
        counter = 0 
        batch_ids = []
        while counter < self.num_samples:
            i = random.sample(range(0, len(os.listdir(self.path))), 1)[0]
            if i not in batch_ids:
                batch_ids.append(i)
                counter += 40
                yield self._open_pickle(i)

    def _open_pickle(self, i):
        with open(self.path + f"batch{i}.bin","rb") as f:
            batch = pickle.load(f)
            for transformation in self.transformations:
                batch = transformation(batch)
            return batch
        
    def load_batch(self, batch_number):
        return self._open_pickle(batch_number)
        
            
    def add_transformation(self, new_transformation):
        self.transformations.append(new_transformation)


