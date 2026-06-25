import pandas as pd
import numpy as np

class AbsorptionSignal:
    def __init__(self):
        pass

    def transform(self, features:pd.DataFrame, horizons:list[int], n_std: int):
        for horizon in horizons:
            features[f"past_z_buy_vol_{horizon}s"] = (features[f"past_buy_vol_{horizon}s"] - features[f"past_buy_vol_{horizon}s"].rolling(10000).mean())\
                /features[f"past_buy_vol_{horizon}s"].rolling(10000).std()
            
            features[f"past_z_sell_vol_{horizon}s"] = (features[f"past_sell_vol_{horizon}s"] - features[f"past_sell_vol_{horizon}s"].rolling(10000).mean())\
                /features[f"past_sell_vol_{horizon}s"].rolling(10000).std()
            
            features[f"past_buy_absorption_{horizon}s"] = np.where((features[f"past_z_sell_vol_{horizon}s"] > n_std) & (features[f"past_mid_return_{horizon}s"]>=0),
                                                                   features[f"past_z_sell_vol_{horizon}s"],
                                                                   0)
            features[f"past_sell_absorption_{horizon}s"] = np.where((features[f"past_z_buy_vol_{horizon}s"] > n_std) & (features[f"past_mid_return_{horizon}s"]<=0),
                                                                   features[f"past_z_buy_vol_{horizon}s"],
                                                                   0)
        return features
