import pandas as pd

from exceptions.TargerBuilderError import *

class TargetBuilder:
    def __init__(self, columns:list[str], lags:list[int]):
       self.columns = columns
       self.lags = lags

    def targetize(self,dataframe:pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:

            if not (column in dataframe.column):
                raise MissingColumnError(f"Column {column} seems to be inexistant.")
            
            for lag in self.lags:
                if lag >= dataframe.size() | lag <= 0:
                    raise InvalidLagError("Lag couldn't be bigger than dataset size or a negative value.")
                colname:str = str(column + "_" + lag)
                dataframe[colname] = dataframe[column].shift(lag)
    
    