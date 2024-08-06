import pandas as pd
from pathlib import Path
import plotly.express as px


def plot_line(workdir: str | Path, metric_name: str, save_path: str | Path = None):
    if isinstance(workdir, str):
        workdir = Path(workdir)
    df_metrics = pd.read_csv(workdir / "metrics.csv")
    assert metric_name in df_metrics, f"Metric {metric_name} not in metrics!"
    fig = px.line(df_metrics, y=metric_name, title=f"Metric {metric_name}")
    fig.update_layout(xaxis_title="Iterations", yaxis_title="Score")
    if save_path is not None:
        save_path = str(save_path)
        if save_path.endswith("html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    return fig
