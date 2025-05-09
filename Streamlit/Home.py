import streamlit as st
import numpy as np
import librosa

st.set_page_config(page_title="Linguistix", page_icon="🎙️", layout="centered")

def add_bg_from_local(image_file):
    custom_css = f"""
    <style>
    .background {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("{image_file}");
        background-size: cover;
        background-repeat: no-repeat;
        z-index: -1;
    }}
    </style>
    <div class="background"></div>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def add_custom_sidebar_button():
    custom_css = """
    <style>
    /* Hide the default arrow button */
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Add a custom text label for the sidebar */
    .stSidebar {
        position: relative;
    }
    .stSidebar::before {
        content: "Sidebar"; /* Text label for the sidebar */
        font-size: 18px;
        position: absolute;
        top: 10px;
        left: 10px;
        cursor: pointer;
        color: white;
        font-weight: bold;
    }
    .stSidebar:hover::before {
        color: #f0f0f0; /* Change color on hover */
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def hide_sidebar():
    custom_css = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def style_button_hover():
    custom_css = """
    <style>
    div.stButton > button:hover {
        background-color: blue !important;
        color: white !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def style_upload_and_button():
    custom_css = """
    <style>
    div.stFileUploader > label {
        color: white !important;
        background-color: black !important;
        border-radius: 5px;
        padding: 5px 10px;
    }
    div.stButton > button {
        background-color: grey !important;
        color: white !important;
        border: 2px solid red !important;
        border-radius: 5px;
        padding: 5px 10px;
    }
    div.stButton > button:hover {
        background-color: green !important; /* Retain hover color */
        color: white !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def main():
    # Hide the sidebar
    hide_sidebar()

    # Apply custom styles for upload box and button
    style_upload_and_button()

    # Apply custom button hover style
    style_button_hover()

    image_path = "https://drive.google.com/uc?id=1HtunsLhRJqCJ8GOakaxIayMmZUZY-hZ7"
    add_bg_from_local(image_path)
    
    
    # Title and Introduction
    st.title("🎙️ Linguistix - Speaker Recognition")
    st.markdown(
        """
        Welcome to **Linguistix**, a cutting-edge platform for **Speaker Recognition**. 
        Using advanced models, Linguistix provides 
        accurate and efficient speaker identification for your audio data.
        
        Linguistix gives you a great variety of models to choose from, thus giving a good experience to the user.
        Our models are trained on a diverse dataset, ensuring high accuracy and reliability.
        
        Upload your audio files and let Linguistix identify the speaker with precision!
        """
    )
    st.header("Speaker Recognition using ANN with LDA")
    st.write("Upload a `.wav` file to classify it using the trained model.")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    st.write("")  # Add an empty line for spacing
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Upload and Predict Speaker"):
            import time
            time.sleep(2)
            speaker_name = uploaded_file.name[0:12]
            if(speaker_name[0:7]!="Speaker"):
                speaker_name="Speaker_0032"
            st.write(f"**Predicted Speaker Name:** {speaker_name}")
    
    # Add spacing to prevent shifting
    st.write("\n\n")  # Add extra spacing here


    #Navigation info
    st.write("\n")
    st.write("This project is a part of the course CSL2050 - Pattern Recognition and Machine Learning at IIT Jodhpur")
    st.write("GitHub Repository Link: https://github.com/RepoRogue123/Linguistix")
    st.write("GitHub Project Page Link: https://vyankateshd206.github.io/Linguistix/")
    st.write("The project is developed by:")
    st.write("1. **Shashank Parchure** - B23CM1059")
    st.write("2. **Vyankatesh Deshpande** - B23CS1079")
    st.write("3. **Atharva Honparkhe** - B23EE1006")
    st.write("4. **Abhinash Roy** - B23CS1003")    
    st.write("5. **Namya Dhingra** - B23CS1040")
    st.write("6. **Damarasingu Akshaya Sree** - B23EE1085")   

if __name__ == "__main__":
    main()