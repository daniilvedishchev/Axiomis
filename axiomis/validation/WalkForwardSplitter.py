from dataclasses import dataclass

import pandas as pd
import numpy as np

from exceptions.WalkForwardError import *

@dataclass
class WalkForwardSplitter:

    train_ratio: float
    validation_ratio: float
    test_ratio: float
    num_buckets: int

    if (train_ratio + validation_ratio + test_ratio != 1):
        raise WalkForwardError("Ratios dont add up to 1.")
    
    def bucketize(self,data_array: np.ndarray) -> list[np.ndarray]:
        bucket_list = list()

        data_size = data_array.shape[0]

        bucket_size = data_size // self.num_buckets
        remainder = data_size % self.num_buckets
        
        bucket_start = 0

        for bucket_idx in range(self.num_buckets):
            bucket_end = bucket_start + bucket_size

            if remainder > 0:
                bucket_end += 1
                remainder -= 1
            
            bucket_list.append(data_array[bucket_start:bucket_end])
            bucket_start = bucket_end

        return bucket_list

            

            

            





    def train_validation_test(self,data_array: np.array) -> list[list]:
        final_array = list()

        initial_bucket_size = data_array.size // self.buckets
        bucket_remainder = data_array.size % self.buckets

        if (bucket_remainder > self.buckets):
            bucket_size = initial_bucket_size + (bucket_remainder // self.buckets)
            bucket_remainder = bucket_remainder % self.buckets
        else:
            bucket_size = initial_bucket_size

        train_bucket_size = round(bucket_size * self.train_ratio)
        validation_bucket_size = round((bucket_size - train_bucket_size) * self.validation_ratio)

        for bucket_idx in range(self.buckets + 1):

            train_start = bucket_idx * bucket_size
            train_validation_test = list()

            if bucket_remainder > 0:
                train_end = train_start + train_bucket_size + 1
                bucket_remainder -= 1
            else:
                train_end = train_start + train_bucket_size
            
            np.append(train_validation_test,data_array[train_start : train_end])

            validation_end = train_end + validation_bucket_size

            train_validation_test.append(data_array[train_end : validation_end])
            train_validation_test.append(data_array[validation_end : -1])

            final_array.append(train_validation_test)

        return final_array
