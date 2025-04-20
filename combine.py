import streamlit as st
from PIL import Image, ImageOps
import pytesseract
import re
import requests
from bs4 import BeautifulSoup
import json

# Set Tesseract command path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def extract_names_from_image(image):
    gray = ImageOps.grayscale(image)
    text = pytesseract.image_to_string(gray)
    names = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text)
    return list(set(names))  # remove duplicates

def compare_attendance(first_image, last_image):
    first_names = extract_names_from_image(first_image)
    last_names = extract_names_from_image(last_image)

    full_attendees = sorted(list(set(first_names) & set(last_names)))
    joined_late = sorted(list(set(last_names) - set(first_names)))
    left_early = sorted(list(set(first_names) - set(last_names)))

    return full_attendees, joined_late, left_early

def extract_leetcode_problems(profile_url):
    try:
        username = profile_url.strip("/").split("/")[-1]
        query = """
        query getUserProfile($username: String!) {
            matchedUser(username: $username) {
                submitStats: submitStatsGlobal {
                    acSubmissionNum {
                        difficulty
                        count
                    }
                }
            }
        }
        """
        variables = {"username": username}
        headers = {
            "Content-Type": "application/json",
            "Referer": f"https://leetcode.com/{username}/",
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.post(
            "https://leetcode.com/graphql",
            json={"query": query, "variables": variables},
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        ac_data = data['data']['matchedUser']['submitStats']['acSubmissionNum']
        problem_counts = {item['difficulty'].lower(): item['count'] for item in ac_data}
        total = sum(problem_counts.values())
        return {
            "easy": problem_counts.get("easy", 0),
            "medium": problem_counts.get("medium", 0),
            "hard": problem_counts.get("hard", 0),
            "total": total,
            "username": username
        }
    except Exception as e:
        return {"error": f"Error accessing LeetCode profile: {e}"}

def count_difference(img1, img2):
    if not img1 or not img2:
        return "Error: Both images are required for comparison."
    size1 = img1.size[0] * img1.size[1]
    size2 = img2.size[0] * img2.size[1]
    return size2 - size1

def main():
    st.set_page_config(page_title="LeetCode & Image Analyzer", page_icon="🔍", layout="wide")

    # CSS for colorful UI and dropdown hover
    st.markdown("""
        <style>
        div.stSelectbox > div[role="combobox"] {
            cursor: pointer;
        }
        div.stSelectbox > div[role="combobox"]:hover,
        div.stSelectbox > div[role="combobox"]:focus-within {
            background-color: #e0f7fa;
        }
        [data-baseweb="base-page"] {
            background-color: #fdf6ff;
        }
        .main-title {
            color: #4a148c;
            text-align: center;
            font-size: 36px;
            padding: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🔍 LeetCode & Image Analyzer</h1>", unsafe_allow_html=True)

    mode = st.sidebar.selectbox(
        "Choose an operation:",
        ["Crowd Count Difference", "LeetCode Profile Analysis", "Attendance Analysis"]
    )

    if mode == "Crowd Count Difference":
        st.header("Crowd Count Difference")
        st.write("Upload two images to compare crowd size (basic estimate).")
        image1 = st.file_uploader("First Image", type=["png", "jpg", "jpeg"], key="img1")
        image2 = st.file_uploader("Second Image", type=["png", "jpg", "jpeg"], key="img2")

        if image1 and image2:
            img1 = Image.open(image1)
            img2 = Image.open(image2)
            st.image(img1, caption="First Image", use_container_width=True)
            st.image(img2, caption="Second Image", use_container_width=True)
            diff = count_difference(img1, img2)
            st.success(f"Estimated crowd difference: {diff}")

    elif mode == "LeetCode Profile Analysis":
        st.header("LeetCode Profile Analysis")
        url = st.text_input("Enter LeetCode profile URL:", "https://leetcode.com/your_username")

        if st.button("Analyze Profile"):
            if "leetcode.com" not in url:
                st.error("Invalid URL.")
            else:
                stats = extract_leetcode_problems(url)
                if "error" in stats:
                    st.error(stats["error"])
                else:
                    st.markdown(f"""
                        <div style="background-color:#ede7f6; padding:1rem; border-radius:10px">
                            <h3 style="color:#6a1b9a;">LeetCode Stats for <strong>{stats['username']}</strong></h3>
                            <ul>
                                <li><strong>Total:</strong> {stats['total']}</li>
                                <li><strong>Easy:</strong> {stats['easy']}</li>
                                <li><strong>Medium:</strong> {stats['medium']}</li>
                                <li><strong>Hard:</strong> {stats['hard']}</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

    elif mode == "Attendance Analysis":
        st.header("Attendance Analysis")
        st.write("Upload images from start and end of the event.")

        first_img = st.file_uploader("Upload FIRST screenshot (beginning of session)", type=["jpg", "jpeg", "png"], key="first")
        last_img = st.file_uploader("Upload LAST screenshot (end of session)", type=["jpg", "jpeg", "png"], key="last")

        if first_img and last_img:
            first_image = Image.open(first_img)
            last_image = Image.open(last_img)

            full, late, early = compare_attendance(first_image, last_image)

            st.success("✅ Attendance Processed")

            st.markdown("### 👨‍💼 Full Attendees")
            st.write(full if full else "None")

            st.markdown("### 🟡 Joined Late")
            st.write(late if late else "None")

            st.markdown("### 🔴 Left Early")
            st.write(early if early else "None")

    # Footer Credit
    st.markdown("""
        <style>
            .creator-footer {
                text-align: center;
                font-size: 18px;
                color: #6a1b9a;
                background-color: #f3e5f5;
                padding: 0.8rem;
                margin-top: 3rem;
                border-radius: 20px;
                box-shadow: 0 -2px 10px rgba(186, 104, 200, 0.3);
            }
        </style>
        <div class="creator-footer">
             🛠️ Created by <strong>HIRANVIKA 🤗🤗</strong>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
