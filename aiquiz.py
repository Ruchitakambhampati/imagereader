import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from sentence_transformers import SentenceTransformer, util
import base64
import os

model = SentenceTransformer('all-MiniLM-L6-v2')


# Ensure Streamlit page config is set first
st.set_page_config(page_title="AI Quiz Generator", page_icon="🚀")

# Initialize session state
if 'form_created' not in st.session_state:
    st.session_state.form_created = False
if 'form_html' not in st.session_state:
    st.session_state.form_html = ""
if 'questions' not in st.session_state:
    st.session_state.questions = []

# Note: Ensure compatible versions: torch>=2.1.0, transformers>=4.40.0, sentence-transformers>=2.2.0
# Install or update with: pip install torch>=2.1.0 transformers>=4.40.0 sentence-transformers>=2.2.0

# Load models with explicit device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "iarfmoose/t5-base-question-generator",
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float32
    )
    model.to(device)  # Ensure model is on the correct device
except Exception as e:
    st.error(f"Failed to load model: {str(e)}. Ensure compatible PyTorch and Transformers versions.")
    st.stop()

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Updated generate_questions function with sampling
def generate_questions(context, num_questions=5):
    prompt = "generate questions: " + context.strip().replace("\n", " ")
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    num_return = num_questions * 3  # Increased to *3 for more candidates
    outputs = model.generate(
        inputs,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=num_return,
        no_repeat_ngram_size=2,
        early_stopping=False,  # Disabled to avoid warning
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = clean_questions(decoded, num_questions)
    return cleaned

def clean_questions(raw_questions, max_questions):
    cleaned = []
    seen = set()
    for q in raw_questions:
        q = q.strip().replace("  ", " ")
        q = re.sub(r"[^\w\d\s?,]", "", q)
        q = re.sub(r"\s+", " ", q).strip("?").strip() + "?"
        q_lower = q.lower()

        if (
            len(q) > 15  # Reduced from 20 to allow shorter questions
            and not re.search(r"\b(name|someone|he|she|him|her|Mr|Ms|Mrs|Dr|Prof)\b", q_lower)
            and q_lower not in seen
        ):
            seen.add(q_lower)
            cleaned.append(q)

    # Semantic deduplication with adjusted threshold
    unique = []
    if cleaned:
        embeddings = similarity_model.encode(cleaned, convert_to_tensor=True)
        for i, q in enumerate(cleaned):
            if len(unique) == 0:
                unique.append(q)
            else:
                sim_scores = util.cos_sim(embeddings[i], embeddings[[cleaned.index(u) for u in unique]])
                if max(sim_scores[0]) < 0.9:  # Changed to 0.9 for less strict deduplication
                    unique.append(q)
            if len(unique) >= max_questions:
                break
    return unique

# Function to create downloadable file
def get_binary_file_downloader_html(bin_file, file_label='File'):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.html">Download {file_label}</a>'
        return href
    return "Error: File not found."

# ------------------ Streamlit App ------------------

st.title("📚 AI Quiz Generator")

st.markdown("Enter your study content or context below, and this app will generate meaningful quiz points.")

context = st.text_area("Enter context or study material here:", height=200)
num_questions = st.slider("Select number of quiz questions to generate:", 1, 20, 5)
form_owner_email = st.text_input("Enter your email to receive form responses:", "")

if st.button("Generate Quiz"):
    if not context.strip():
        st.warning("Please enter some content to generate questions.")
    elif not form_owner_email:
        st.warning("Please enter your email to receive form responses.")
    else:
        with st.spinner("Generating questions..."):
            try:
                questions = generate_questions(context, num_questions)
                if questions:
                    st.session_state.questions = questions
                    st.success("✨ Here are your quiz questions:")
                    for i, q in enumerate(questions, 1):
                        st.markdown(f"**{i}. {q}**")

                    # Ask if questions are okay
                    if st.button("Are these questions okay?"):
                        if st.button("Yes"):
                            # Ask if user wants to create a form
                            if st.button("Can I create a form?"):
                                # Create Formspree-compatible HTML form
                                formspree_endpoint = f"https://formspree.io/f/{form_owner_email}"  # Placeholder; replace with actual Formspree URL
                                form_html = f"""
                                <html>
                                <head><title>Quiz Form</title></head>
                                <body>
                                <h1>Quiz Form</h1>
                                <form action="{formspree_endpoint}" method="POST">
                                {''.join([f'<div><label>{i}. {q}</label><input type="text" name="q{i}" required></div>' for i, q in enumerate(st.session_state.questions, 1)])}
                                <input type="hidden" name="total_questions" value="{len(st.session_state.questions)}">
                                <input type="hidden" name="points_per_question" value="1">
                                <input type="submit" value="Submit">
                                </form>
                                <p>Download this file, replace the Formspree endpoint with your unique URL from <a href="https://formspree.io">Formspree</a>, and share it. Responses will be emailed to {form_owner_email}.</p>
                                </body>
                                </html>
                                """
                                st.markdown("### Form Created!")
                                st.markdown("### Form HTML Preview:")
                                st.code(form_html, language="html")
                                st.success(f"Each question is worth 1 point. Total points possible: {len(st.session_state.questions)}")
                                st.markdown("**Note:** Sign up at Formspree to get your unique endpoint and replace the placeholder URL.")

                                # Save form to a temporary file and set session state
                                with open("quiz_form.html", "w") as f:
                                    f.write(form_html)
                                st.session_state.form_html = form_html
                                st.session_state.form_created = True
                else:
                    st.warning("Couldn't generate valid questions. Try a different input.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display Download Form button if form is created
if st.session_state.form_created:
    st.write("Form is ready for download!")
    if st.button("Download Form"):
        with open("quiz_form.html", "w") as f:
            f.write(st.session_state.form_html)
        st.markdown(get_binary_file_downloader_html("quiz_form.html", "Quiz Form"), unsafe_allow_html=True)
        st.markdown("Click the link above to download the form.")
else:
    st.markdown("Generate, confirm questions, and create a form to enable the Download Form button.")