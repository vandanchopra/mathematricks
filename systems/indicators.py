import pandas as pd

class Indicators():
    def __init__(self,data=None):
        self.data = pd.DataFrame()

    def get_all_indicators(self,indicators):
        self.indicator_data = pd.DataFrame()
        for indicator in indicators:
            if "sma" in indicator.lower():
                try:
                    n = int(indicator.lower().replace("sma",""))
                    self.sma(n)
                except Exception as e:
                    raise Exception(f"Wrong Data Type for SMA {e}")


    def sma(self,n):
        sma_data = self.data.rolling(window=n).mean()
        sma_data.columns = pd.MultiIndex.from_tuples([(f"{col[0]}_SMA{n}",col[1]) for col in sma_data.columns])
        self.indicator_data = pd.concat([self.data,self.indicator_data,sma_data],axis = 1)