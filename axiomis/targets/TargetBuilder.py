import pandas as pd

from exceptions.TargerBuilderError import *
from enum import Enum


class TargetMode(Enum):
    FUTURE_VALUE = "future_value"
    RETURN = "return"
    BINARY = "binary"
    MULTICLASS = "multiclass"

class TargetBuilder:
    def __init__(self, columns:list[str], lags:list[int],modes:list[TargetMode], fee:float, spread_bps:int):
       self.columns = columns
       self.lags = lags
       self.modes = modes
       self.fee = fee
       self.spread_bps = spread_bps

       self.TARGET_FUNCTIONS = {
            TargetMode.RETURN: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag) / dataframe[column] - 1
            ),

            TargetMode.FUTURE_VALUE: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag)
            ),

            TargetMode.BINARY: lambda dataframe, column, lag: (
                self.TARGET_FUNCTIONS[TargetMode.RETURN](dataframe, column, lag)
                > self.spread_bps + self.fee
            ),
        }

    def targetize(self,dataframe:pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:

            if not (column in dataframe.columns):
                raise MissingColumnError(f"Column {column} seems to be inexistant.")
            
            for lag in self.lags:
                if lag >= len(dataframe) or lag <= 0:
                    raise InvalidLagError("Lag couldn't be bigger than dataset size or a negative value.")
                for mode in self.modes:
                    if (mode == TargetMode.RETURN):
                        colname:str = f"{column}_lag_{lag}"
                colname:str = f"{column}_lag_{lag}"
                dataframe[colname] = dataframe[column].shift(-lag)/dataframe[column] - 1
    
    