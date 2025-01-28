import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.stats import chi2
import plotly.io as pio
import os
from weasyprint import HTML
from io import BytesIO
import streamlit as st
from io import StringIO
import itertools
import plotly.express as px
import matplotlib.pyplot as plt
from tueplots import axes, bundles

###########################################################################################################
#file handling stuff, parsing

def handle_file_input(file_input):
    if isinstance(file_input, str) and os.path.isfile(file_input):
        return open(file_input, 'r') 
    elif hasattr(file_input, 'getvalue'):
        return StringIO(file_input.getvalue().decode("utf-8"))  
    else:
        raise ValueError("Input must be either a file path or an uploaded file object.")


def parse_geno(input_data):
    geno_lines = []
    try:
        with handle_file_input(input_data) as file:
            geno_lines = file.readlines()
    except Exception as e:
        raise ValueError("Input must be either a file path or a file-like object.") from e

    geno_data = [list(map(int, line.strip())) for line in geno_lines]
    geno_array = np.array(geno_data, dtype=np.uint8).T
    return np.where(geno_array == 9., np.nan, geno_array) 

def parse_ind(file_input):
    try:
        with handle_file_input(file_input) as file:
            ind_df = pd.read_csv(file, delim_whitespace=True, header=None, names=['ID', 'Gender', 'Population'], engine='python')
    except Exception as e:
        raise ValueError("Input must be either a file path or a file-like object.") from e

    return ind_df


def create_subset(input_geno, input_ind, selected_indices, output_geno_path, output_ind_path):
    # Handle IND file
    ind_file = handle_file_input(input_ind)
    with ind_file, open(output_ind_path, 'w') as out_ind_file:
        for idx, line in enumerate(ind_file):
            if idx in selected_indices:
                out_ind_file.write(line)

    # Handle GENO file
    geno_file = handle_file_input(input_geno)
    with geno_file, open(output_geno_path, 'w') as out_geno_file:
        for line in geno_file:
            selected_columns = ''.join(line[i] for i in selected_indices)
            out_geno_file.write(selected_columns + "\n")

###########################################################################################################
def missing_statistics(geno, ind):
    # creates dict from number of missing genotype positions indexed by ID
    total_positions = geno.shape[1]
    missing_data_percentage = {
        ind.iloc[i].values[0]: (np.isnan(geno[i, :]).sum() / total_positions) * 100
        for i in range(geno.shape[0])
    }
    return missing_data_percentage, total_positions

def get_nonvariant_geno(geno, indices):
    print("get nonv feno")
    # filters genos only for variant positions
    # TODO: more general!
    indices = indices["x"].values - 1
    return geno[:, indices]

def plot_missing(nines):
    # creates bar plot of missing percentage per sample
    sample_names = list(nines.keys())
    missing_percentage = list(nines.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sample_names,
        y=missing_percentage,
        marker=dict(color='lightsalmon'),
        name="Missing Data (%)",
        hoverinfo="x+y",
    ))
    fig.add_trace(go.Scatter(
        x=sample_names,
        y=[100] * len(sample_names),
        mode="lines",
        line=dict(color="indianred", dash="dash"),
        name="100% (Total Positions)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Missing Data per Sample",
        xaxis_title="Samples",
        yaxis_title="Missing Data (%)",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        height=500,
        width=1200,
        showlegend=True,
    )
    return fig

def compute_tau(geno, genomean, V, is_not_nani):
    snp_drift = np.sqrt((genomean / 2) * (1 - genomean / 2))
    geno_norm = (geno - genomean) / snp_drift
    V_obs = V[is_not_nani]
    proj_factor = np.linalg.inv(V_obs[:, 0:2].T @ V_obs[:, 0:2]) @ V_obs[:, 0:2].T
    tau = proj_factor @ geno_norm[is_not_nani]
    return tau

def pmp_drift_parallel(genos, V, genomean, is_not_nan, n_jobs=-1):
    taus = Parallel(n_jobs=n_jobs)(
        delayed(compute_tau)(geno, genomean, V, is_not_nan[i]) for i, geno in enumerate(genos)
    )
    return taus

def save_fig_as_pdf(fig):
    pdf_buffer = BytesIO()
    fig.write_image(pdf_buffer, format="pdf", engine="kaleido")
    pdf_buffer.seek(0) 
    return pdf_buffer

def color_plot(modern_df, taus, inds):
    modern_df = modern_df.sort_values(by=['Group'])
    markers = ['circle', 'triangle-up', 'square', 'pentagon', 'star', 'diamond', 'diamond-wide', 'hexagon', 'x']
    palette = px.colors.qualitative.Vivid
    palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

    style = list(itertools.product(palette, markers))
    modern_df['Style'] = modern_df['Group'].map(dict(zip(modern_df['Group'].unique(),
                                       style)))
    fig = go.Figure()

    for sty in modern_df['Style'].unique():
        df_sub = modern_df.loc[modern_df['Style'] == sty]
        fig.add_trace(
            go.Scatter(
                x=df_sub['PC1'],
                y=df_sub['PC2'],
                mode='markers',
                marker=dict(
                    size=5,
                    symbol=sty[1],  # Marker-Stil (z. B. 'o', '^', etc.)
                    color=f"rgb({sty[0][0]*255},{sty[0][1]*255},{sty[0][2]*255})",  # Farbe in RGB
                    line=dict(width=0.1, color='white')  # Kantenfarbe
                ),
                name=df_sub['Group'].unique()[0]  # Gruppenname als Legende
            )
        )

    # Layout anpassen
    fig.update_layout(
        width=488,
        height=int(0.75 * 488),
        title="Modern Samples",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        legend=dict(title="Group")
    )

    for t, tau in enumerate(taus): 
        name = inds.iloc[t].values[0]
        sample_text = f"{name}"
        # Sample hinzufügen
        fig.add_trace(go.Scatter(
            x=[tau[0]],
            y=[tau[1]],
            mode='markers+text',
            marker=dict(color='black', size=8),
            name=name,
            text=sample_text,
            textposition="top center",
            legendgroup=name,
        ))

    fig.update_layout(
        width=1300,
        height=1050,
        xaxis=dict(
        showgrid=False,
        zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white',
        legend=dict(
            #title="Modern West Eurasians",
            itemsizing='constant',  
            traceorder="normal",  
            orientation="h",
            x=0, #x=1.05,
            y=-0.6, #y=0.5, 
            xanchor="left",
            yanchor="bottom", 
            font=dict(
                size=12,
                family="Arial" 
            ),
            itemwidth=120,
            title_font=dict(
                size=12
            ),
            bordercolor="white",  
            borderwidth=0.2 
            ),
            margin=dict(l=80, r=80, t=0, b=20)
    )

    return fig


def base_plot(modern, taus, inds, color_lookup):
    fig = go.Figure()

    # Modern samples hinzufügen
    coords_mwe = modern[["PC1", "PC2"]]

    fig.update_layout(
        width=1300,
        height=1050,
        xaxis=dict(
        showgrid=False,
        zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white',
        xaxis_title="PC1",
        yaxis_title="PC2"

    )
    fig.add_trace(go.Scatter(
        x=coords_mwe["PC1"],
        y=coords_mwe["PC2"],
        mode='markers',
        marker=dict(color='rgb(127, 127, 127)'),
        name='Modern Samples',
    ))

    for t, tau in enumerate(taus): 
        name = inds.iloc[t]["ID"]
        sample_text = f"{name}"
        # Sample hinzufügen
        fig.add_trace(go.Scatter(
            x=[tau[0]],
            y=[tau[1]],
            mode='markers+text',
            marker=dict(color=color_lookup[name], size=10),
            name=name,
            text=sample_text,
            textposition="top center",
            legendgroup=name
        ))

    return fig

def get_ellipse(mean, Sigma, confidence_level):
    chi2_val = chi2.ppf(confidence_level, df=2)
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    # Sorting eigenvalues and corresponding eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Width and height of the ellipse (2 * sqrt(eigenvalue * chi-square value))
    width = 2 * np.sqrt(eigvals[0] * chi2_val)
    height = 2 * np.sqrt(eigvals[1] * chi2_val)

    # Angle of the ellipse in degrees (in the direction of the largest eigenvector)
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    x_center, y_center = mean
   
    t = np.linspace(0, 2 * np.pi, 100)
    x = (width / 2) * np.cos(t)
    y = (height / 2) * np.sin(t)

    # Rotation 
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    # Translation to center
    x_final = x_rot + x_center
    y_final = y_rot + y_center

    return x_final, y_final
    #return ellipse

def var_discrepency(V_obs, var_tau_r):
  
  #Computes the variance discrepancy between the estimated embedding and the true embedding.
  matrix_of_linear_map = - np.linalg.inv(V_obs[:, 0:2].T @ V_obs[:, 0:2]) @ V_obs[:, 0:2].T @ V_obs[:, 2:]
  return matrix_of_linear_map @ var_tau_r @ matrix_of_linear_map.T

def set_active_tab(tab_name):
    st.session_state["active_tab"] = tab_name

# Callback-Funktion zur Synchronisation
def update_percentiles():
    st.session_state["selected_percentiles"] = st.session_state["percentile_selector"]

def create_pdf_report(html_text, plotly_figures): #s, table_html):
    # Plots als SVG exportieren und in HTML einfügen
    plot_svgs = []
    for fig in plotly_figures:
        svg_data = pio.to_image(fig, format="svg", width=600, height=400)
        plot_svgs.append(svg_data.decode("utf-8"))  # SVG-Daten als Text

    # Komplette HTML-Struktur für das PDF
    full_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .table th, .table td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            .table th {{
                background-color: #f2f2f2;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        {html_text}
        <h2>Plots</h2>
    """

    # Füge Plots als SVG ein
    for i, plot_svg in enumerate(plot_svgs):
        full_html += f"""
        <h3>Plot {i + 1}</h3>
        <div>{plot_svg}</div>
        """

    # Füge Tabelle ein
    #full_html += f"""
    #    <h2>Tabelle</h2>
    #    {table_html}
    #</body>
    #</html>
    #"""

    # HTML in PDF umwandeln
    pdf = HTML(string=full_html).write_pdf()
    return BytesIO(pdf)

