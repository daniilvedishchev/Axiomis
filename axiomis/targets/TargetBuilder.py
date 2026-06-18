import pandas as pd
import numpy as np

from enum import Enum
from exceptions.TargerBuilderError import *
from axiomis.targets.helpers.Helper import *

class TargetMode(Enum):
    FUTURE_VALUE = "future_value"
    RETURN = "return"
    BINARY = "binary"
    MULTICLASS = "multiclass"
    DELTA = "delta"

class TargetBuilder:
    def __init__(self, columns:list[str], lags:list[int]):
       self.columns = columns
       self.lags = lags

       self.COLUMN_TARGET_CONFIG = {
           "mid": {
               "mode": TargetMode.RETURN,
               "threshold": "fraction"
           },
           "microprice_l1": {
               "mode": TargetMode.RETURN,
               "threshold": "fraction"
           },
           "relative_microprice_bps": {
               "mode": TargetMode.DELTA,
               "threshold": "bps"
           }
       }

       self.TARGET_FUNCTIONS = {
            TargetMode.RETURN: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag) / dataframe[column] - 1
            ),

            TargetMode.FUTURE_VALUE: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag)
            ),

            TargetMode.DELTA: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag) - dataframe[column]
            )
        }
       

    def targetize(self,dataframe:pd.DataFrame) -> pd.DataFrame:
        targets = pd.DataFrame()
        for column in self.columns:
            if not (column in dataframe.columns):
                raise MissingColumnError(f"Column {column} seems to be inexistant.")
            
            for lag in self.lags:
                if lag >= len(dataframe) or lag <= 0:
                    raise InvalidLagError("Lag couldn't be bigger than dataset size or a negative value.")
                

                config = self.COLUMN_TARGET_CONFIG[column]
                target_values = self.TARGET_FUNCTIONS[config["mode"]](dataframe,column,lag)


                targets[f"{column}_lag_{lag}_mode_{config['mode'].value}"] = target_values
        return targets
    
    