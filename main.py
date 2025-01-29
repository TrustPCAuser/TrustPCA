import pandas as pd
import numpy as np
import streamlit as st
import copy
import pickle
import tempfile

import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.stats import chi2
import plotly.io as pio
import os
# from weasyprint import HTML
from io import BytesIO
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
        template="simple_white",
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
        template="simple_white",
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
    pdf = None #HTML(string=full_html).write_pdf()
    return BytesIO(pdf)

app_description = """
    <h1>Information</h1>
    <p>
        This platform provides statistics and visualization to assess the uncertainty of genotype sample projections in a Principal Component Analysis (PCA).<br>
        Derived from the SmartPCA algorithm, the placement variability of ancient genomic data points in the feature space is quantified and displayed based on modern West-Eurasian samples.
    </p>

    <h2>Data</h2>
    <p>
        To assess the uncertainty in your PCA placement, please upload the following data:
    </p>
    <ul>
        <li><b>GENO-Datei</b>: The genotype data in EIGENSTRAT format (other formats will be supported soon)</li>
        <li><b>IND-Datei</b>: Information on the individuals (e.g. name, population).</li>
        <li><b>SNP-Datei</b>: The SNP names and positions.</li>
    </ul>
    """
background_files = """
    <style>
    .file-upload-section {
        background-color: lightsalmon; 
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """

top_margin = """
    <style>
    .title {
        position: relative;
        top: -50px; /* Verschiebt den Titel nach oben */
        text-align: center; /* Zentriere den Titel */
        font-size: 48px; /* Schriftgröße */
        color: darkblue; /* Standardfarbe des Titels */
    }
    .highlight {
        color: indianred; /* Farbe für das Wort "PCA" */
    }
    .ellipses {
        position: relative;
        top: -60px; /* Verschiebe die Ellipsen-Grafik nach oben */
        text-align: center; /* Zentriere die Grafik */
    }
    .description {
        position: relative;
        top: -58px;
        text-align: center;
        font-size:32px !important;
        color: black;
    }
    </style>
    """
alert = """
    <style>
    .stAlert {
        background-color: lightsalmon !important;
        border-radius: 10px; /* Optionale abgerundete Ecken */
        padding: 10px; /* Innenabstand */
    }
    </style>
    """
title =  """
    <h1 class="title">
        Welcome to TRUST<span class="highlight">PCA</span>!
    </h1>
    """

ellipse_logo_old = """
    <div class="ellipses">
        <svg width="500" height="50" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="250" cy="25" rx="100" ry="5" fill="red" opacity="0.7" />
            <ellipse cx="250" cy="25" rx="200" ry="8" fill="red" opacity="0.4" />
            <circle cx="250" cy="25" r="5" fill="blue" />
        </svg>
    </div>
    """

ellipse_logo = """
    <div class="ellipses">
        <svg width="500" height="50" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="250" cy="25" rx="100" ry="5" fill="lightsalmon" opacity="0.7" />
            <ellipse cx="250" cy="25" rx="200" ry="8" fill="lightsalmon" opacity="0.4" />
            <circle cx="250" cy="25" r="5" fill="tomato" />
        </svg>
    </div>
    """

subtitle =  """
    <p class="description">
        A <span class="highlight">T</span>ool for <span class="highlight">R</span>eliability and <span class="highlight">U</span>ncertainty in <span class="highlight">S</span>mar<span class="highlight">T</span>PCA projections
    </p>
    """

buttons =  """
    <style>
    div.stButton>button {
       font-size: 100px !important; /* Schriftgröße */
        padding: 15px 30px !important; /* Abstand innerhalb der Buttons */
        height: 60px !important; /* Mindesthöhe der Buttons */
        width: 200px !important; /* Mindestbreite der Buttons */
        margin: 10px !important; /* Abstand zwischen den Buttons */
        border-radius: 8px !important; /* Abgerundete Ecken */
        background-color: white !important; /* Hintergrundfarbe */
        color: black !important; /* Schriftfarbe */
        font-weight: bold !important; /* Fettere Schrift */
    }
    div.stButton>button:hover {
        background-color: lightblue; /* Hover-Farbe */
        color: black; /* Schriftfarbe beim Hover */
    }
    </style>
    """

buttons= """
<style>
div.stButton>button {
    font-size: 20px !important; /* Einheitliche Schriftgröße */
    padding: 20px !important; /* Einheitlicher Abstand innerhalb der Buttons */
    border-radius: 8px !important; /* Abgerundete Ecken */
    margin: 0 5px !important; /* Abstand zwischen den Tabs */
    background-color: #e0e0e0 !important; /* Hintergrundfarbe */
    color: black !important; /* Schriftfarbe */
    font-weight: bold !important; /* Fettere Schrift */
    border: 2px solid #ccc !important; /* Standard Rahmen */
    width: 200px !important; /* Einheitliche Breite */
    height: 70px !important; /* Einheitliche Höhe */
    text-align: center !important; /* Zentrierung des Inhalts */
    display: inline-block !important; /* Buttons in einer Reihe */
    transition: all 0.2s ease-in-out; /* Sanfte Animation */
}

div.stButton>button:hover {
    background-color: #d6d6d6 !important; /* Hover-Hintergrund */
    color: black !important; /* Hover-Schriftfarbe */
}
</style>
"""

############################################################
# Standard files 

#path_to_database = "../../ancientPCA/database/"
path_to_database = "./database/"
modern = pd.read_csv(path_to_database+'coordinates_MWE.csv')
modern_df = pd.read_csv(path_to_database+'embedding_modern_refs.csv')
groups = pd.read_csv(path_to_database+'modern_groups_curated.csv', header=0)
modern_df['Group'] = groups['Group']
indices = pd.read_csv(path_to_database+'SNPs_mwe.csv', header=0)
genomean = pd.read_csv(path_to_database+'genomean.csv', header=0)
genomean = genomean['x'].values
V = np.load(path_to_database+'eigenvectors.npy')
Lambda = np.load(path_to_database+'eigenvalues.npy')
magical_factors = np.load(path_to_database+'factors.npy')

############################################################
# set globally 
percentiles = [0.99, 0.9, 0.75, 0.5]
base_palette = px.colors.qualitative.Vivid
  

# App Layout
st.set_page_config(layout="wide",page_title="TRUST PCA")

#smaller top margin 
st.markdown(top_margin, unsafe_allow_html=True)

# title
st.markdown(title, unsafe_allow_html=True
)

#Logo
st.markdown(
    ellipse_logo,
    unsafe_allow_html=True
)

#subtitle
st.markdown(
   subtitle,
    unsafe_allow_html=True
)

# Initialize vars
geno = None
ind = None

# Initialize session state
if "geno" not in st.session_state:
    st.session_state["geno"] = None
if "ind" not in st.session_state:
    st.session_state["ind"] = None
if "nines" not in st.session_state:
    st.session_state["nines"] = None
if "taus" not in st.session_state:
    st.session_state["taus"] = None
if "ellipses" not in st.session_state:
    st.session_state["ellipses"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Genotype Data" 
if "selected_percentiles" not in st.session_state:
    st.session_state["selected_percentiles"] = percentiles
if "missing_plot" not in st.session_state:
    st.session_state["missing_plot"] = None
if "ellipse_plot" not in st.session_state:
    st.session_state["ellipse_plot"] = None
if "sample_submitted" not in st.session_state:
    st.session_state["sample_submitted"] = False
if "preprocessing" not in st.session_state:
    st.session_state["preprocessing"] = False
if "checkbox_states" not in st.session_state:
    st.session_state["checkbox_states"] = []
if "select_all" not in st.session_state:
    st.session_state["select_all"] = False
if "example_data" not in st.session_state:
    st.session_state["example_data"] = False
if "parsed" not in st.session_state:
    st.session_state["parsed"] = False
if "color_lookup" not in st.session_state:
    st.session_state["color_lookup"] = False

st.markdown(
    background_files,
    unsafe_allow_html=True
)

st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# file uploads
st.header("Please upload your EIGENSTRAT files or use Example Data")
st.write("Upload your GENO and IND files to begin processing or use example data to test the application.")

col1, col2 = st.columns(2)

with col1:
    geno_file = st.file_uploader("Upload GENO file", type=["csv", "txt", "geno"])

with col2:
    ind_file = st.file_uploader("Upload IND file", type=["csv", "txt", "ind"])

if st.button("Use Example Data"):
    st.session_state["example_data"] = True

status_banner = st.empty()
tabs = ["Information"]

if st.session_state["example_data"]:
    st.info("Using example data...")
    ancient_example = pd.read_csv(path_to_database+"example_data_tool.csv", header=0)
    with open(path_to_database+"ellipses_example_data.pkl", "rb") as f:
        ellipses = pickle.load(f)
        st.session_state["ellipses"] = ellipses
    nines = ancient_example[["ID","coverage"]]
    nines["coverage"]=nines["coverage"]/594924 * 100
    nines_dict = nines.set_index("ID")["coverage"].to_dict()
    st.session_state["nines"]=nines_dict
    st.session_state["missing_plot"] = plot_missing(st.session_state["nines"])
    st.session_state["ind"] = ancient_example[["ID", "Group_ID"]]
    st.session_state["taus"] = np.array(ancient_example[["x", "y"]])
    
    num_inds = len(ancient_example["ID"])
    scaled_palette = [base_palette[i % len(base_palette)] for i in range(num_inds)]
    color_lookup = dict(zip(ancient_example["ID"], scaled_palette))
    st.session_state["color_lookup"] = color_lookup
    fig = base_plot(modern, st.session_state["taus"], st.session_state["ind"], color_lookup)
    color_plot = color_plot(modern_df, st.session_state["taus"], st.session_state["ind"])
    st.session_state["base_plot"] = fig
    st.session_state["color_plot"] = color_plot
    st.session_state["preprocessing"] = True
       

if geno_file and ind_file:
    ind_data = parse_ind(ind_file)
    if len(st.session_state["checkbox_states"]) != len(ind_data):
        st.session_state["checkbox_states"] = [False] * len(ind_data)
    st.subheader("Select Individuals for Analysis")
    
    st.session_state["select_all"] = st.checkbox("Select All Individuals", value=st.session_state["select_all"])
    if st.session_state["select_all"]:
        st.session_state["checkbox_states"] = [True] * len(ind_data)

    num_columns = 3
    columns = st.columns(num_columns)

    for i, row in ind_data.iterrows():
        col_index = i % num_columns
        with columns[col_index]:
            entry = f"{row['ID']} - {row['Gender']} - {row['Population']}"
            st.session_state["checkbox_states"][i] = st.checkbox(
                entry, value=st.session_state["checkbox_states"][i], key=f"checkbox_{i}"
            )

    if st.button("Submit Selection"):
        selected_indices = [i for i, selected in enumerate(st.session_state["checkbox_states"]) if selected]
        if selected_indices:
            st.session_state["sample_submitted"] = True
            if not st.session_state.select_all:
                with tempfile.TemporaryDirectory(dir=".") as temp_dir:
                    output_geno_path = f"{temp_dir}/subset_geno.geno"
                    output_ind_path = f"{temp_dir}/subset_ind.ind"
                    create_subset(geno_file, ind_file, selected_indices, output_geno_path, output_ind_path)
                    st.session_state["geno"] = parse_geno(output_geno_path)
                    st.session_state["ind"] = parse_ind(output_ind_path)
            else:
                st.session_state["geno"] = parse_geno(geno_file)
                st.session_state["ind"] = ind_data
            st.session_state["parsed"]=True


if st.session_state["parsed"]:
    print("sample submitted")
    if not st.session_state["preprocessing"]:
        status_banner.info("Filtering genos...")
        nonv_geno = get_nonvariant_geno(st.session_state["geno"], indices)
        st.session_state["geno"] = nonv_geno
        status_banner.info("Calculating missing statistics...")
        nines, total_positions = missing_statistics(st.session_state["geno"], st.session_state["ind"])
        st.session_state["nines"] = nines
        st.session_state["missing_plot"] = plot_missing(st.session_state["nines"])
        is_not_nan = ~np.isnan(nonv_geno)
        status_banner.info("Projecting samples...")
        taus = pmp_drift_parallel(nonv_geno, V, genomean, is_not_nan, n_jobs=-1) #pmp_drift(nonv_geno, V, genomean, is_not_nan)
        st.session_state["taus"] = taus
        status_banner.info("Generating PCA plot...")
        num_inds=st.session_state["ind"].shape[0]
        scaled_palette = [base_palette[i % len(base_palette)] for i in range(num_inds)]
        color_lookup = dict(zip(st.session_state["ind"]["ID"], scaled_palette))
        st.session_state["color_lookup"] = color_lookup
        fig = base_plot(modern, taus, st.session_state["ind"], color_lookup)
        st.session_state["base_plot"] = fig
        color_plot = color_plot(modern_df, st.session_state["taus"], st.session_state["ind"])
        
        st.session_state["color_plot"] = color_plot

        ellipses = {} #ellipses can not be a df bc arrow compatibility or sth
        status_banner.info("Calculating uncertainties...") 

        progress_bar = st.progress(0)  # Start bei 0%
        total_samples = len(nonv_geno)
        var_tau_r = np.diag(Lambda[2:] * magical_factors[2:])
        for index, geno in enumerate(nonv_geno):
            curr_ind = st.session_state["ind"].iloc[index]
            progress = (index + 1) / total_samples
            progress_bar.progress(progress)

            ellipses[curr_ind["ID"]] = {}
        
            is_not_nani = is_not_nan[index]
            # Predict variance of discrepency
            V_obs = V[is_not_nani]
            var_discr = var_discrepency(V_obs=V_obs, var_tau_r=var_tau_r)
            #ellipses
            for percentile in percentiles:
                x, y = get_ellipse(mean=taus[index], Sigma=var_discr, confidence_level=percentile)
                ellipses[curr_ind["ID"]][percentile] = {"x": x, "y": y}

        progress_bar.empty()
        status_banner.empty()
        st.session_state["ellipses"] = ellipses
        status_banner.empty()
        st.session_state["preprocessing"] = True

st.markdown(
    background_files,
    unsafe_allow_html=True
)
st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["preprocessing"]:
    st.write(app_description, unsafe_allow_html=True,)


if st.session_state["preprocessing"]:
    tabs = ["Genotype Data", "Uncertainty Analysis", "Information"]
    st.markdown(
    buttons,
    unsafe_allow_html=True,
    )

    columns = st.columns(len(tabs))

    # TODO: design of buttons 
    for col, tab_name in zip(columns, tabs):
        with col:
            if st.button(tab_name, key=f"button_{tab_name}"):
                set_active_tab(tab_name)


    # Tab 1
    if st.session_state["active_tab"] == "Genotype Data":
        st.header("Genotype Data")
        #st.success("GENO and IND data uploaded successfully!")
        if st.session_state["sample_submitted"]:
            with st.expander("Data Preview", expanded=False):
                st.subheader("Data preview")
                st.write("Geno (First 50x50)")
                st.write(st.session_state["geno"][0:50, 0:50])
                st.write("Ind (First Rows)")
                st.write(st.session_state["ind"].head())
        st.subheader("Missing Data Statistics")
        st.write("Missing Data per Sample (in %)")
        nines_df = pd.DataFrame.from_dict(st.session_state["nines"], orient="index", columns=["Missing SNP Percentage"])
        nines_df["Missing SNP Percentage"] = nines_df["Missing SNP Percentage"].round(2).apply(lambda x: f"{x:.2f}")
        st.table(nines_df)
        if st.session_state["missing_plot"]:
            st.plotly_chart(st.session_state["missing_plot"])
        st.subheader("Sample Placement Based on SmartPCA")
        
        if "current_plot" not in st.session_state:
            st.session_state["current_plot"] = "base"  # Standardmäßig den Base Plot anzeigen

        # Button zum Wechseln des Plots
        if st.button("Change Coloring"):
            # Wechsel zwischen "base" und "color"
            if st.session_state["current_plot"] == "base":
                st.session_state["current_plot"] = "color"
            else:
                st.session_state["current_plot"] = "base"

        # Plot basierend auf dem Zustand anzeigen
        if st.session_state["current_plot"] == "base":
            st.plotly_chart(st.session_state["base_plot"], use_container_width=True)
        elif st.session_state["current_plot"] == "color":
            st.plotly_chart(st.session_state["color_plot"], use_container_width=True)
        

    # Tab 2
    elif st.session_state["active_tab"] == "Uncertainty Analysis":
        st.header("Uncertainty Analysis")
        # change design muslitselect
        selected = st.multiselect(
        label="Select Percentiles",
        options=[0.99, 0.9, 0.75, 0.5],
        default=st.session_state["selected_percentiles"],
        key="percentile_selector",
        on_change=update_percentiles
        )      
        if st.button("Change Coloring"):
            st.session_state["current_plot"] = "base" if st.session_state["current_plot"] == "color" else "color"

        if st.session_state["current_plot"] == "base":
            fig = copy.deepcopy(st.session_state["base_plot"])
        else:
            fig = copy.deepcopy(st.session_state["color_plot"])


        alpha = 0.2
        for index, ind_i in st.session_state["ind"].iterrows():
            color_rgb = st.session_state["color_lookup"][ind_i["ID"]][4:-1].split(",")  # Entferne 'rgb(' und ')'
            fillcolor = f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, {alpha})'
            for percentile in st.session_state["selected_percentiles"]:
                coords = st.session_state["ellipses"][ind_i["ID"]][percentile]
                fig.add_trace(go.Scatter(
                    x=np.append(coords["x"], coords["x"][0]),  # Ellipse schließen
                    y=np.append(coords["y"], coords["y"][0]),
                    mode="none",
                    fill="toself",
                    fillcolor=fillcolor if st.session_state["current_plot"] == "base" else f'rgba(0, 0, 0, {alpha})',
                    legendgroup=ind_i["ID"], 
                    showlegend=False,
                    name=f"{percentile}"
                ))
        

        #sadly, the tau traces have to be added again, bc they are overwritten by the ellipses and should go to the foreground (no better way)
        #if st.session_state["current_plot"] == "base":
        #    for trace in st.session_state["base_plot"].data:
        #        if trace.legendgroup in st.session_state["ind"]["ID"].values:
        #            fig.add_trace(trace)
        #else:
        #    for trace in st.session_state["color_plot"].data:
        #        if trace.legendgroup in st.session_state["ind"]["ID"].values:
        #            fig.add_trace(trace)
    
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["ellipse_plot"] = fig

        pdf_buffer = save_fig_as_pdf(st.session_state["ellipse_plot"])
        st.download_button(
            label="Download Figure as PDF",
            data=pdf_buffer,
            file_name="TRUST_PCA_download.pdf",
            mime="application/pdf"
        )
        
        st.header("PDF-Report")
        st.markdown("To download the Summary of Missing statistics and Uncertainty Plots, please click the Download-Button")
        pdf_buffer = create_pdf_report(
            html_text=app_description,
            plotly_figures=[st.session_state["missing_plot"], st.session_state["ellipse_plot"]]
            #table_html=table_html
        )
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="Uncertainty_analysis_report.pdf",
            mime="application/pdf"
        )


    # Tab 3
    elif st.session_state["active_tab"] == "Information":
        st.markdown(app_description, unsafe_allow_html=True)

