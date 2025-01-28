import pandas as pd
import numpy as np
import streamlit as st
import copy
from functions import *
from styles import *
import pickle
import tempfile

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

