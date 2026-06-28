import pandas as pd
import numpy as np
import re

class FeatureEngine:
    def __init__(self):
        self.depth: list[int] = [10,20,50,100,200,500,1000]
        self.time_shifts: list[int] = [60,300,600]

    def shift_imbalance_features(self, features:pd.DataFrame, shifts:list[int]):

        imbalance_dict = {}
        features_copy = features.copy()
        imbalances: list[int] = [imbalance for imbalance in features.columns if "imb_" in imbalance]

        for imbalance in imbalances:
            for shift in shifts:
                imbalance_dict[f"past_{imbalance}_{shift}s"] = (features_copy[imbalance]/features_copy[imbalance].shift(shift))-1
                imbalance_dict[f"{imbalance}_delta_{shift}s"] = features_copy[imbalance] - imbalance_dict[f"past_{imbalance}_{shift}s"]
        
        features_upd = pd.concat([features,pd.DataFrame(imbalance_dict,index=features.index)],axis=1)
        

        return features_upd
    
    def weight_imbalance_deltas(self, features: pd.DataFrame) -> pd.DataFrame:
        
        columns_to_weight = [
            column for column in features.columns
            if re.fullmatch(r"imb_\d+", column)
        ]

        columns_to_weight = sorted(
            columns_to_weight,
            key=lambda x: int(x.split("_")[1])
        )

        weights = np.exp(np.arange(len(columns_to_weight)))
        weights = weights / weights.sum()
        weights = weights[::-1]

        weighted_deltas_dict = {}

        for time_shift in self.time_shifts:
            weighted_deltas_dict[f"weighted_delta_imbalance_{time_shift}s"] = np.sum(
                [
                    weights[i] * features[f"{column}_delta_{time_shift}s"]
                    for i, column in enumerate(columns_to_weight)
                ],
                axis=0,
            )

        return pd.concat(
            [features, pd.DataFrame(weighted_deltas_dict, index=features.index)],
            axis=1
        )

                



