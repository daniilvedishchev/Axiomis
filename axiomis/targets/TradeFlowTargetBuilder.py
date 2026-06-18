import pandas as pd
import numpy as np

from axiomis.targets.helpers.Helper import *

class TradeFlowTargetBuilder:
    def __init__(self,trades_dataframe:pd.DataFrame,):
        pass
    
    def calculate_forward_trade_flow_base_features_by_horizon(
    trades_dataframe: pd.DataFrame,
    reference_timestamps_ms: np.array,
    horizons_sec: list[int],
    time_col: str = "trade_time_ms"
) -> np.ndarray:

        is_first_window: bool = True

        trade_timestamps = trades_dataframe[time_col].to_numpy(dtype=int)
        trade_quantities = trades_dataframe["quantity"].to_numpy(dtype=float)
        trade_side = trades_dataframe["is_buyer_maker"].to_numpy(dtype=int)
        trade_prices = trades_dataframe["price"].to_numpy(dtype=int)

        n_trades = len(trade_timestamps)
        n_orderbook = len(reference_timestamps_ms)

        order_quantities_by_horizon = np.array(
            [np.zeros(2, dtype=float) for _ in horizons_sec]
        )
        right_ptr_by_horizon = np.zeros(len(horizons_sec), dtype=int)

        features = ["buy_vol","sell_vol","buy_count","sell_count","total_vol","vwap_buy","vwap_sell"]

        buy_volume_result      = np.zeros((len(horizons_sec), n_orderbook))
        sell_volume_result     = np.zeros((len(horizons_sec), n_orderbook))
        buy_count_result       = np.zeros((len(horizons_sec), n_orderbook))
        sell_count_result      = np.zeros((len(horizons_sec), n_orderbook))
        total_volume_result    = np.zeros((len(horizons_sec), n_orderbook))
        vwap_buy_result        = np.zeros((len(horizons_sec), n_orderbook))
        vwap_sell_result       = np.zeros((len(horizons_sec), n_orderbook))

        buy_qty_by_horizon     = np.zeros(len(horizons_sec))
        sell_qty_by_horizon    = np.zeros(len(horizons_sec))
        buy_count_by_horizon   = np.zeros(len(horizons_sec))
        sell_count_by_horizon  = np.zeros(len(horizons_sec))
        vwap_num_by_horizon    = np.zeros(len(horizons_sec))
        vwap_buy_by_horizon     = np.zeros(len(horizons_sec))
        vwap_sell_by_horizon     = np.zeros(len(horizons_sec))

        left_ptr = 0
        right_ptr = 0
        previous_left_pointer = 0

        buy_qty = 0.0
        buy_count = 0
        sell_qty = 0.0
        sell_count = 0
        vwap_num = 0
        vwap_buy = 0
        vwap_sell = 0

        for anchor_idx in range(n_orderbook):

            while (
                left_ptr < n_trades
                and reference_timestamps_ms[anchor_idx] > trade_timestamps[left_ptr]
            ):
                left_ptr += 1

            if is_first_window:
                right_ptr = left_ptr

            for horizon_idx, horizon_s in enumerate(horizons_sec):

                if not is_first_window:
                    buy_qty = buy_qty_by_horizon[horizon_idx]
                    sell_qty = sell_qty_by_horizon[horizon_idx]
                    buy_count = buy_count_by_horizon[horizon_idx]
                    sell_count = sell_count_by_horizon[horizon_idx]
                    vwap_num = vwap_num_by_horizon[horizon_idx]
                    vwap_buy = vwap_buy_by_horizon[horizon_idx]
                    vwap_sell = vwap_sell_by_horizon[horizon_idx]

                    if left_ptr - previous_left_pointer > 0:
                        for i in range(previous_left_pointer, left_ptr):
                            if trade_side[i] == 0:
                                buy_qty -= trade_quantities[i]
                                buy_count -= 1
                                vwap_buy -= trade_quantities[i] * trade_prices[i]
                            else:
                                sell_qty -= trade_quantities[i]
                                sell_count -= 1
                                vwap_sell -= trade_quantities[i] * trade_prices[i]

                    right_ptr = right_ptr_by_horizon[horizon_idx]

                while (
                    right_ptr < n_trades
                    and trade_timestamps[right_ptr]
                    < reference_timestamps_ms[anchor_idx] + to_ms(horizon_s)
                ):
                    if trade_side[right_ptr] == 0:
                        buy_qty += trade_quantities[right_ptr]
                        buy_count += 1
                        vwap_buy += trade_quantities[right_ptr] * trade_prices[right_ptr]
                    else:
                        sell_qty += trade_quantities[right_ptr]
                        sell_count += 1
                        vwap_sell += trade_quantities[right_ptr] * trade_prices[right_ptr]

                    right_ptr += 1

                right_ptr_by_horizon[horizon_idx] = right_ptr

                total_qty = buy_qty + sell_qty

                buy_volume_result[horizon_idx][anchor_idx] = buy_qty
                sell_volume_result[horizon_idx][anchor_idx] = sell_qty
                buy_count_result[horizon_idx][anchor_idx] = buy_count
                sell_count_result[horizon_idx][anchor_idx] = sell_count
                total_volume_result[horizon_idx][anchor_idx] = total_qty
                vwap_buy_result[horizon_idx][anchor_idx] = vwap_buy/buy_qty if buy_qty>0 else 0
                vwap_sell_result[horizon_idx][anchor_idx] = vwap_sell/sell_qty if sell_qty>0 else 0

                if is_first_window:
                    for j in range(len(horizons_sec)):
                        right_ptr_by_horizon[j] = right_ptr
                        buy_qty_by_horizon[j]     = buy_qty
                        sell_qty_by_horizon[j]    = sell_qty
                        buy_count_by_horizon[j]   = buy_count
                        sell_count_by_horizon[j]  = sell_count
                        vwap_num_by_horizon[j]    = vwap_num
                        vwap_buy_by_horizon[j]     = vwap_buy
                        vwap_sell_by_horizon[j]     = vwap_sell

                    is_first_window = False
                else:
                    buy_qty_by_horizon[horizon_idx]     = buy_qty
                    sell_qty_by_horizon[horizon_idx]    = sell_qty
                    buy_count_by_horizon[horizon_idx]   = buy_count
                    sell_count_by_horizon[horizon_idx]  = sell_count
                    vwap_num_by_horizon[horizon_idx]    = vwap_num
                    vwap_buy_by_horizon[horizon_idx]     = vwap_buy
                    vwap_sell_by_horizon[horizon_idx]     = vwap_sell

            previous_left_pointer = left_ptr
        
        result_vec = np.array([buy_volume_result,sell_volume_result,buy_count_result,sell_count_result,total_volume_result,vwap_buy_result,vwap_sell_result])

        
        feature_dataframe = {f"{feature}_{horizon}s": result_vec[i][idx] for i,feature in enumerate(features) for idx,horizon in enumerate(horizons_sec)}
        feature_dataframe["ts_provider_ms"] = reference_timestamps_ms

        return pd.DataFrame(feature_dataframe)