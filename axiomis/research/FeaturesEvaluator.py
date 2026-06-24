import pandas as pd
import numpy as np

class FeatureEvaluator:
    def __init__(self):
        pass

    def feature_importance(self, features: pd.DataFrame, targets: pd.DataFrame, n_top: int) -> pd.DataFrame:
        correlation_table: pd.DataFrame = pd.DataFrame(index=features.columns,columns=targets.columns)
        for idx,target_column in enumerate(targets):
            if ("ms" in target_column):
                continue
            if ("mode_return" in target_column) and ("mid_" in target_column):
                correlation_table[target_column] = features.corrwith(targets[target_column])
        
        leaderboard = correlation_table.stack().reset_index()
        leaderboard.columns = ["feature","target","corr"]

        leaderboard["abs_corr"] = leaderboard["corr"].abs()
        leaderboard = leaderboard.sort_values(by="abs_corr",ascending=False).iloc[:n_top]
        leaderboard.index = [i for i in range(len(leaderboard))]
        
        return leaderboard
    
    def decile_analysis(self, features_table:pd.DataFrame, targets_table:pd.DataFrame, correlation_leader_table:pd.DataFrame, monotonicity_threshold:float) -> pd.DataFrame:
        updated_corr_table = correlation_leader_table.copy()

        updated_corr_table["decile_spread"] = np.zeros(len(correlation_leader_table))
        updated_corr_table["SNR"] = np.zeros(len(correlation_leader_table))

        for i,(feature,target) in enumerate(zip(correlation_leader_table["feature"],correlation_leader_table["target"])):
            if ("absorption" in feature):
                continue
            temporary: pd.DataFrame = pd.DataFrame({"feature":features_table[feature],"target":targets_table[target]})
            temporary["buckets"] = pd.qcut(temporary["feature"],10,labels=False)

            decile = temporary.groupby("buckets")["target"].mean()

            diff = decile.diff(1)
            diff.dropna(inplace=True)
            monotonicity_coef = max((diff > 0).mean(),(diff < 0).mean())

            if (monotonicity_coef >= monotonicity_threshold):
                updated_corr_table.loc[i,"decile_spread"] = decile.iloc[-1] - decile.iloc[0]
                updated_corr_table.loc[i,"SNR"] = updated_corr_table.loc[i,"decile_spread"]/features_table[feature].std()
            else:
                updated_corr_table.loc[i,"decile_spread"] = pd.NA
        return updated_corr_table




        