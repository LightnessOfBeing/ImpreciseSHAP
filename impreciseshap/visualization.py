from typing import List, Any, Optional

import pandas as pd
import plotly.graph_objs as go

from impreciseshap import ImpreciseShap

FONT_SIZE = 22
FONT_COLOR = 'black'


def get_df_for_eps(model: Any, df_train: pd.DataFrame, df_points_to_explain: pd.DataFrame,
                   eps_arr: Optional[List[float]] = None):
    if eps_arr is None:
        eps_arr = [1e-3, 1e-2, 5e-2, 0.1, 0.15]
    df_stack = None
    res_eps_arr = []
    for epsilon in eps_arr:
        ms = ImpreciseShap(model=model.predict_proba, masker=df_train, eps=epsilon)
        df_res = ms.calculate_shapley_values(df_points_to_explain)
        if df_stack is None:
            df_stack = df_res
        else:
            df_stack = pd.concat([df_stack, df_res], axis=0)
        res_eps_arr.extend([epsilon] * len(df_points_to_explain))
    df_stack = df_stack.reset_index(drop=True)
    df_stack['eps'] = res_eps_arr
    return df_stack


def encode_eps(x: float) -> int:
    if x == 1e-3:
        return 1
    elif x == 1e-2:
        return 2
    elif x == 5e-2:
        return 3
    elif x == 0.1:
        return 4
    return 5


def plot_eps(model: Any, df_train: pd.DataFrame, df_points_to_explain: pd.DataFrame,
             eps_arr: Optional[List[float]] = None, y_range=None):
    if eps_arr is None:
        eps_arr = [1e-3, 1e-2, 5e-2, 0.1, 0.15]

    if y_range is None:
        y_range = [-0.2, 0.6]
    df_points_to_explain = df_points_to_explain.reset_index(drop=True)
    df_eps = get_df_for_eps(model, df_train, df_points_to_explain, eps_arr)
    df_points_to_explain['eps_encoded'] = df_eps['eps'].apply(lambda x: encode_eps(x))
    for i in range(len(df_points_to_explain)):
        fig = go.Figure()
        xsl = []
        ysl = []
        df_grouped = df_grouped.reset_index(drop=True)
        for i in df_grouped.index:
            ysl.append(df_grouped.iloc[i]['mean l'])
            ysl.append(df_grouped.iloc[i]['mean u'])

            xsl.append(df_grouped.iloc[i]['eps_encoded'])
            xsl.append(df_grouped.iloc[i]['eps_encoded'])

        s1 = go.Scatter(
            x=xsl,
            y=ysl,
            mode='markers',
            marker=dict(symbol=symbol, size=symbol_size, color='rgb(0, 0, 255)')
        )
        fig.add_trace(s1)

        for i in df_grouped.index:
            t = go.Scatter(
                x=[df_grouped.iloc[i]['eps_encoded']] * 2,
                y=[df_grouped.iloc[i]['mean l'], df_grouped.iloc[i]['mean u']],
                marker=dict(symbol=symbol, size=symbol_size, color='rgb(0, 0, 255)'))
            fig.add_trace(t)

        fig.update_layout(
            font=dict(
                size=FONT_SIZE,
                color=FONT_COLOR
            ),
            showlegend=False, width=600, height=600,
            margin=dict(l=margin + 75, r=margin,
                        t=margin, b=margin)
        )
        fig.update_layout(
            xaxis=dict(
                title=r'$\huge{\epsilon}$',
                range=[0.5, 5.5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['0.001', '0.01', '0.05', '0.1', '0.15'],
                title_standoff=ts
            ),
            yaxis=dict(
                title=fr"$\huge{{\phi({f_name})}}$",
                range=y_range,
                title_standoff=ts + 6,
            )
        )
        fig.show()
