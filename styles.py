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