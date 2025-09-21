st.markdown(
    """
    <style>
    /* Fullscreen dark background */
    .stApp {
        background: linear-gradient(160deg, #050505, #121212 90%) !important;
        color: #f5f5f5 !important;
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px; /* larger base font */
    }
    .main {
        background: transparent !important;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 3em 1em 2em 1em;
    }
    .hero h1 {
        font-size: 3.8em !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #ff4b4b, #ffaaaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3em;
    }
    .hero p {
        color: #e0e0e0 !important;
        font-size: 1.3em !important;
        font-weight: 500 !important;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 2.5em;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.95);
        backdrop-filter: blur(14px);
        transition: transform 0.3s ease;
        font-size: 1.2em !important;
        font-weight: 600 !important;
    }
    .glass-card:hover {
        transform: scale(1.02);
    }

    /* Sliders */
    div[data-baseweb="slider"] > div {
        height: 16px !important;
        background: rgba(255,255,255,0.25);
        border-radius: 14px;
    }
    div[data-baseweb="slider"] span {
        height: 32px !important;
        width: 32px !important;
        background: #ff4b4b !important;
        border: 3px solid white !important;
        border-radius: 50%;
        box-shadow: 0px 0px 20px #ff4b4b;
    }
    label {
        font-size: 1.2em !important;
        font-weight: 600 !important;
        color: #f0f0f0 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #b22222);
        color: white !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        padding: 1em 1.8em;
        border-radius: 16px;
        border: none;
        width: 100%;
        transition: all 0.25s ease-in-out;
        box-shadow: 0px 5px 25px rgba(255,75,75,0.6);
    }
    .stButton > button:hover {
        transform: scale(1.08);
        box-shadow: 0px 8px 30px rgba(255,75,75,0.9);
    }

    /* Result card */
    .result-card {
        padding: 2.5em;
        margin: 2.5em auto;
        border-radius: 22px;
        text-align: center;
        font-size: 2em !important;
        font-weight: 800 !important;
        width: 80%;
        animation: fadeIn 0.8s ease-in-out;
    }
    .good {
        background: rgba(0, 128, 0, 0.2);
        color: #98fb98 !important;
        border: 3px solid #32cd32;
        box-shadow: 0px 0px 25px rgba(50,205,50,0.8);
    }
    .bad {
        background: rgba(178,34,34,0.2);
        color: #ff9999 !important;
        border: 3px solid #ff4b4b;
        box-shadow: 0px 0px 25px rgba(255,75,75,0.8);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 1.1em !important;
        margin: 3em 0 1em 0;
        color: #aaa !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
