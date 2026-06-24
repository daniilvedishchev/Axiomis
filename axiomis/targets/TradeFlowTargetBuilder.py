import pandas as pd
import numpy as np

from axiomis.targets.helpers.Helper import to_ms


class TradeFlowBaseBuilder:
    def __init__(self, horizons_s: list[int]):
        self.horizons_sec = sorted(horizons_s)

    def _add_derived(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()

        for h in self.horizons_sec:
            buy_vol = df[f"{prefix}_buy_vol_{h}s"]
            sell_vol = df[f"{prefix}_sell_vol_{h}s"]
            buy_count = df[f"{prefix}_buy_count_{h}s"]
            sell_count = df[f"{prefix}_sell_count_{h}s"]
            vwap_buy = df[f"{prefix}_vwap_buy_{h}s"]
            vwap_sell = df[f"{prefix}_vwap_sell_{h}s"]

            total_vol = buy_vol + sell_vol
            total_count = buy_count + sell_count

            df[f"{prefix}_trade_imbalance_{h}s"] = np.where(
                total_vol > 0,
                (buy_vol - sell_vol) / total_vol,
                0.0,
            )

            df[f"{prefix}_cvd_{h}s"] = buy_vol - sell_vol

            df[f"{prefix}_trade_count_imbalance_{h}s"] = np.where(
                total_count > 0,
                (buy_count - sell_count) / total_count,
                0.0,
            )

            df[f"{prefix}_avg_buy_size_{h}s"] = np.where(
                buy_count > 0,
                buy_vol / buy_count,
                0.0,
            )

            df[f"{prefix}_avg_sell_size_{h}s"] = np.where(
                sell_count > 0,
                sell_vol / sell_count,
                0.0,
            )

            df[f"{prefix}_vwap_spread_{h}s"] = vwap_buy - vwap_sell

        return df

    def _build_window_flow(
        self,
        trades_dataframe: pd.DataFrame,
        reference_timestamps_ms: np.ndarray,
        direction: str,
        prefix: str,
        time_col: str = "trade_time_ms",
    ) -> pd.DataFrame:
        trades = trades_dataframe.copy().sort_values(time_col)

        trade_ts = trades[time_col].to_numpy(dtype=np.int64)
        qty = trades["quantity"].to_numpy(dtype=float)
        side = trades["is_buyer_maker"].to_numpy(dtype=int)
        price = trades["price"].to_numpy(dtype=float)

        n_trades = len(trade_ts)
        n_refs = len(reference_timestamps_ms)
        n_h = len(self.horizons_sec)

        left_ptr = np.zeros(n_h, dtype=int)
        right_ptr = np.zeros(n_h, dtype=int)

        buy_qty = np.zeros(n_h)
        sell_qty = np.zeros(n_h)
        buy_count = np.zeros(n_h)
        sell_count = np.zeros(n_h)
        buy_pq = np.zeros(n_h)
        sell_pq = np.zeros(n_h)

        results = {
            f"{prefix}_buy_vol_{h}s": np.zeros(n_refs) for h in self.horizons_sec
        }

        for h in self.horizons_sec:
            results[f"{prefix}_sell_vol_{h}s"] = np.zeros(n_refs)
            results[f"{prefix}_buy_count_{h}s"] = np.zeros(n_refs)
            results[f"{prefix}_sell_count_{h}s"] = np.zeros(n_refs)
            results[f"{prefix}_total_vol_{h}s"] = np.zeros(n_refs)
            results[f"{prefix}_vwap_buy_{h}s"] = np.zeros(n_refs)
            results[f"{prefix}_vwap_sell_{h}s"] = np.zeros(n_refs)

        for ref_idx, anchor_ms in enumerate(reference_timestamps_ms):
            for h_idx, h in enumerate(self.horizons_sec):
                h_ms = to_ms(h)

                if direction == "forward":
                    window_start = anchor_ms
                    window_end = anchor_ms + h_ms
                elif direction == "past":
                    window_start = anchor_ms - h_ms
                    window_end = anchor_ms
                else:
                    raise ValueError("direction must be 'forward' or 'past'")

                while left_ptr[h_idx] < n_trades and trade_ts[left_ptr[h_idx]] < window_start:
                    i = left_ptr[h_idx]

                    if side[i] == 0:
                        buy_qty[h_idx] -= qty[i]
                        buy_count[h_idx] -= 1
                        buy_pq[h_idx] -= qty[i] * price[i]
                    else:
                        sell_qty[h_idx] -= qty[i]
                        sell_count[h_idx] -= 1
                        sell_pq[h_idx] -= qty[i] * price[i]

                    left_ptr[h_idx] += 1

                while right_ptr[h_idx] < n_trades and trade_ts[right_ptr[h_idx]] < window_end:
                    i = right_ptr[h_idx]

                    if side[i] == 0:
                        buy_qty[h_idx] += qty[i]
                        buy_count[h_idx] += 1
                        buy_pq[h_idx] += qty[i] * price[i]
                    else:
                        sell_qty[h_idx] += qty[i]
                        sell_count[h_idx] += 1
                        sell_pq[h_idx] += qty[i] * price[i]

                    right_ptr[h_idx] += 1

                total = buy_qty[h_idx] + sell_qty[h_idx]

                results[f"{prefix}_buy_vol_{h}s"][ref_idx] = buy_qty[h_idx]
                results[f"{prefix}_sell_vol_{h}s"][ref_idx] = sell_qty[h_idx]
                results[f"{prefix}_buy_count_{h}s"][ref_idx] = buy_count[h_idx]
                results[f"{prefix}_sell_count_{h}s"][ref_idx] = sell_count[h_idx]
                results[f"{prefix}_total_vol_{h}s"][ref_idx] = total

                results[f"{prefix}_vwap_buy_{h}s"][ref_idx] = (
                    buy_pq[h_idx] / buy_qty[h_idx] if buy_qty[h_idx] > 0 else 0.0
                )

                results[f"{prefix}_vwap_sell_{h}s"][ref_idx] = (
                    sell_pq[h_idx] / sell_qty[h_idx] if sell_qty[h_idx] > 0 else 0.0
                )
        results["ts_provider_ms"] = reference_timestamps_ms
        base_df = pd.DataFrame(results)
        return self._add_derived(base_df, prefix)


class TradeFlowTargetBuilder(TradeFlowBaseBuilder):
    def build_forward_targets(
        self,
        trades_dataframe: pd.DataFrame,
        reference_timestamps_ms: np.ndarray,
        time_col: str = "trade_time_ms",
    ) -> pd.DataFrame:
        return self._build_window_flow(
            trades_dataframe=trades_dataframe,
            reference_timestamps_ms=reference_timestamps_ms,
            direction="forward",
            prefix="forward",
            time_col=time_col,
        )


class TradeFlowFeatureBuilder(TradeFlowBaseBuilder):
    def build_past_features(
        self,
        trades_dataframe: pd.DataFrame,
        reference_timestamps_ms: np.ndarray,
        time_col: str = "trade_time_ms",
    ) -> pd.DataFrame:
        return self._build_window_flow(
            trades_dataframe=trades_dataframe,
            reference_timestamps_ms=reference_timestamps_ms,
            direction="past",
            prefix="past",
            time_col=time_col,
        )