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

class TargetBuilder:
    def __init__(self, columns:list[str], lags:list[int], modes:list[TargetMode], fee_bps:int, spread_bps:list[int]):
       self.columns = columns
       self.lags = lags
       self.modes = modes
       self.fee_bps = fee_bps
       self.spread_bps = spread_bps

       self.TARGET_FUNCTIONS = {
           TargetMode.RETURN: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag) / dataframe[column] - 1
            ),

            TargetMode.FUTURE_VALUE: lambda dataframe, column, lag: (
                dataframe[column].shift(-lag)
            ),

            TargetMode.MULTICLASS: lambda dataframe, column, lag, spread: (
                lambda returns, threshold: np.where(
                    returns.isna(),np.nan,
                    np.where(returns > threshold,1,
                            np.where(returns < -threshold,-1,0)
                    )
                )
            )(self.TARGET_FUNCTIONS[TargetMode.RETURN](dataframe, column, lag),bps_to_fraction(spread + self.fee_bps))
        }
       

    def targetize(self,dataframe:pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:

            if not (column in dataframe.columns):
                raise MissingColumnError(f"Column {column} seems to be inexistant.")
            
            for lag in self.lags:
                if lag >= len(dataframe) or lag <= 0:
                    raise InvalidLagError("Lag couldn't be bigger than dataset size or a negative value.")
                for mode in self.modes:
                    colname:str = f"{column}_lag_{lag}_mode_{mode.value}"
                    if (mode == TargetMode.MULTICLASS):
                        for spread in self.spread_bps:
                            colname_with_spread = colname + f"_spread_bps_{spread}"
                            dataframe[colname_with_spread] = self.TARGET_FUNCTIONS[mode](dataframe,column,lag,spread)
                    else : 
                        dataframe[colname] = self.TARGET_FUNCTIONS[mode](dataframe,column,lag)
        return dataframe
    
    