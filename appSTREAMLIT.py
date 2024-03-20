import streamlit as st
import os
import subprocess
import sys
import threading
import  pyrebase
import datetime



def save_uploaded_file(uploaded_file, save_dir, filename):
    with open(os.path.join(save_dir, filename), "wb") as f:
        f.write(uploaded_file.getbuffer())

def run_ml(progress_bar):
    python_executable = sys.executable
    subprocess.run([python_executable, "main.py"])
    st.success("ok")


def load_value():
    try:
        # Load the current value from the text file
        with open('variable_value.txt', 'r') as file:
            current_value = int(file.read())
    except FileNotFoundError:
        # If the file doesn't exist, initialize with a default value
        current_value = 5
    return current_value

def update_value(new_value):
    # Update the value in the text file
    with open('variable_value.txt', 'w') as file:
        file.write(str(int(new_value)))

def main():
    st.title("Neural Style Transfer")
    st.write("Neural Style Transfer is a technique that employs deep neural networks, such as VGG19, to blend the artistic style of one image with the content of another, resulting in visually appealing synthesized images.")
    firebaseConfig = {
    'apiKey': "AIzaSyAhCKUMbP8GGtwkAaAEV38dKWzn6BcxS5Y",
    'authDomain': "neural-st.firebaseapp.com",
    'projectId': "neural-st",
    'storageBucket': "neural-st.appspot.com",
    'messagingSenderId': "690387462569",
    'appId': "1:690387462569:web:8b5709267710c33a227cb8",
    'databaseURL': "https://neural-st-default-rtdb.firebaseio.com/" }
    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()

    db = firebase.database()
    storage = firebase.storage()

    #st.sidebar("login")
    option = st.radio("Select Image Source:", ("Take Picture", "Upload from System"))

    if option == "Take Picture":
        picture = st.camera_input("Take a picture:")
        if picture:
            with open('content/contentpic.jpg', 'wb') as file:
                file.write(picture.getbuffer())
            st.success("Picture taken successfully!")

    elif option == "Upload from System":
        content_uploaded_file = st.file_uploader("Upload Content Image", type=None)
        if content_uploaded_file is not None:
            save_dir_content = "content"
            if not os.path.exists(save_dir_content):
                os.makedirs(save_dir_content)
            save_uploaded_file(content_uploaded_file, save_dir_content, "contentpic.jpg")
            st.success("Content image uploaded successfully!")
            st.image("content/contentpic.jpg", caption="Content Image")
    
    style_uploaded_file = st.file_uploader("Upload Style Image", type=None)
    
    if style_uploaded_file is not None:
        save_dir_style = "style"
        if not os.path.exists(save_dir_style):
            os.makedirs(save_dir_style)
        save_uploaded_file(style_uploaded_file, save_dir_style, "stylepic.jpg")
        st.success("Style image uploaded successfully!")
        st.image("style/stylepic.jpg", caption="Style Image")

    current_value = load_value()
    new_value = st.slider('Adjust the strength:', 1, 10, current_value)

    if new_value != current_value:
        update_value(new_value)
        st.success('Value updated successfully!')

    if st.button("Transfer Style"):
        st.write("Processing....")
        progress_bar = st.progress(0)  # Create progress bar
        thread = threading.Thread(target=run_ml, args=(progress_bar,))
        thread.start()
        thread.join()  # Wait for the thread to finish
        st.success("Image Generated.")

        if os.path.exists("generated_image.jpg"):
            st.image("generated_image.jpg", caption="Generated Image")
        else:
            st.error("No generated image found!")

if __name__ == "__main__":
    main()