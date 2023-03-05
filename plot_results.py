import argparse
import json
import plotly.graph_objs as go
import plotly.io as pio

# Plots the evolution of expected cumulative regrets curves,
# for all tested policies and over all rounds
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="results.json", required=False,
                        help="path to data")

    args = parser.parse_args()

    with open(args.data_path, 'r') as fp:
        cumulative_regrets = json.load(fp)

    fig = go.Figure()

    for k,v in cumulative_regrets.items():
        fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, name=k))

    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Cumulative Regret",
        legend=dict(font=dict(size=18)),
        title = {
            'text': "Batch Size = 30000",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        width=800,  # set width to 800 pixels (half of a 1600-pixel screen)
        height=600,  # set height to 600 pixels
    )
    pio.show(fig)
