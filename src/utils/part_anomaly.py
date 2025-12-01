import pandas as pd


def part_anomaly_fractions(
    csv_path: str,
    *,
    sp_col: str = "spatter_px",
    st_col: str = "streak_px",
    total_col: str = "total_px",
    pid_col: str = "part_id",
) -> pd.DataFrame:
    """Aggregate anomaly fractions per printed part from the CSV."""

    df = pd.read_csv(csv_path)

    required = {sp_col, st_col, total_col, pid_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Sum the three pixel columns per part
    agg = df.groupby(pid_col, as_index=True).agg(
        total_px=pd.NamedAgg(column=total_col, aggfunc="sum"),
        spatter_px=pd.NamedAgg(column=sp_col, aggfunc="sum"),
        streak_px=pd.NamedAgg(column=st_col, aggfunc="sum"),
    )

    # Combine & compute fraction
    agg["anomaly_px"] = agg["spatter_px"] + agg["streak_px"]
    agg["anomaly_frac"] = agg["anomaly_px"] / agg["total_px"]

    # Return sorted, tidy frame
    return agg[["anomaly_frac", "total_px", "anomaly_px"]].sort_values("anomaly_frac")
