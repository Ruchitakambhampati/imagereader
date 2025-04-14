import streamlit as st
import torch
from PIL import Image, ImageEnhance
from io import BytesIO
from datetime import datetime
import pytesseract
import base64
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Configure Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except FileNotFoundError:
    st.error("Tesseract OCR is not installed or the path is incorrect. Please install it and update the path.")

# Image captioning function
def analyze_image_locally(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Preprocessing
def preprocess_image(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2)

# Convert image to base64 string
def prepare_image_for_api(image: Image.Image) -> str:
    try:
        buffered = BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        buffered.seek(0)
        return base64.b64encode(buffered.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# App config
st.set_page_config(page_title="Image Understanding Chatbot", layout="wide")
st.title("🖼 Image Understanding Chatbot")
st.caption("Upload an image and ask questions about it")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select model",
        ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"],
        index=0
    )
    max_tokens = st.slider("Max response length", 100, 2000, 500)

    st.divider()
    st.header("Task Mode Controls")

    if "task_mode" not in st.session_state:
        st.session_state.task_mode = False
        st.session_state.task_conversation = []
        st.session_state.task_images = []

    if not st.session_state.task_mode:
        if st.button("Enter Task Mode"):
            st.session_state.task_mode = True
            st.session_state.task_conversation = []
            st.session_state.task_images = []
            st.rerun()
    else:
        st.success("Task Mode Active")
        if st.button("Generate Report & Exit Task Mode"):
            st.session_state.task_mode = False
            st.session_state.show_report = True
            st.rerun()

    st.divider()
    st.header("Search History")
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    for idx, item in enumerate(st.session_state.search_history[::-1]):
        with st.expander(f"Query {len(st.session_state.search_history)-idx}: {item['timestamp']}"):
            st.caption(f"Question: {item['question']}")
            st.write(f"Response: {item['response'][:150]}...")

    if st.button("Clear History"):
        st.session_state.search_history = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            if message.get("image"):
                st.image(message["image"], caption="Uploaded Image", use_column_width=True)
            st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# Show report if available
if hasattr(st.session_state, "show_report") and st.session_state.show_report:
    st.subheader("📄 Task Mode Report")

    report_content = "## Task Mode Session Report\n\n"
    report_content += f"*Session Date:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report_content += "### Conversation Summary\n"
    for idx, msg in enumerate(st.session_state.task_conversation):
        role = "User" if msg["role"] == "user" else "Assistant"
        report_content += f"{role} {idx+1}:** {msg['content']}\n\n"

    if st.session_state.task_images:
        report_content += "### Image Analysis Summary\n"
        for idx, img in enumerate(st.session_state.task_images):
            report_content += f"*Image {idx+1}:* [Attached in report]\n"

    st.markdown(report_content)

    st.download_button(
        label="Download Report",
        data=report_content,
        file_name=f"task_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

    if st.button("Close Report"):
        st.session_state.show_report = False
        st.rerun()

# Main interaction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
user_question = st.chat_input("Ask a question about the image...")

if uploaded_file is not None and user_question:
    try:
        image = Image.open(uploaded_file)
        image = preprocess_image(image)
        base64_image = prepare_image_for_api(image)

        with st.chat_message("user"):
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(user_question)

        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
            "image": "data:image/jpeg;base64," + base64_image if base64_image else None
        })

        if base64_image is None:
            raise Exception("Image encoding failed")

        with st.spinner("Analyzing image..."):
            assistant_response = analyze_image_locally(image)

    except Exception as e:
        assistant_response = f"❌ Error: {str(e)}"
        if "model" in str(e) and "not found" in str(e):
            assistant_response = "❌ Error: The selected model is not available. Please choose a different model."

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })

    if st.session_state.task_mode:
        st.session_state.task_conversation.append({
            "role": "user",
            "content": user_question
        })
        if image not in st.session_state.task_images:
            st.session_state.task_images.append(image)
        st.session_state.task_conversation.append({
            "role": "assistant",
            "content": assistant_response
        })

    st.session_state.search_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": user_question,
        "response": assistant_response,
        "images": ["data:image/jpeg;base64," + base64_image if base64_image else None],
        "full_conversation": st.session_state.messages.copy()
    })