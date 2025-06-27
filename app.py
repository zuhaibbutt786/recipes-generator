import streamlit as st
from PIL import Image
from ultralytics import YOLO
import requests
import json
import tempfile

# Load model
model = YOLO("bestmodel.pt")  # Replace with 'best.pt' if you have a custom trained model

# Function to detect vegetables/fruits from image
def detect_vegetables(image_path):
    results = model(image_path)
    boxes = results[0].boxes
    class_names = results[0].names

    detected_items = set()
    for box in boxes:
        class_id = int(box.cls[0])
        label = class_names[class_id]
        detected_items.add(label.lower())

    return list(detected_items)

# Function to create prompt for Mistral

def build_recipe_prompt(detected_items):
    items = ", ".join(detected_items)
    return f"""Act as a Pakistani expert chef.

I have the following ingredients available: {items}.

Suggest 3 traditional or creative Pakistani recipes using these ingredients.  
Each recipe should include:
- Name of the dish
- Required ingredients (quantities optional)
- Step-by-step cooking instructions

Use local Pakistani cooking styles, spices, and techniques. Avoid non-Pakistani dishes."""

# Function to call Mistral API
def call_mistral_api(prompt):
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
        "X-Title": "VeggieRecipeBot",
        "HTTP-Referer": "http://localhost"
    }

    payload = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI Setup
st.set_page_config(page_title="🥗 Smart Recipe Assistant", layout="centered")
st.title("🥗 Smart Recipe Assistant")
st.markdown("Upload a fruit/vegetable image or take a photo from your mobile to get delicious Pakistani recipes! 🧑\u200d🍳")

# Upload or capture image
image_file = st.file_uploader("Take a photo or upload an image", type=["jpg", "png", "jpeg"])

if "detected_items" not in st.session_state:
    st.session_state.detected_items = set()

if image_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(image_file.read())
    image_path = tfile.name

    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting ingredients..."):
        detected = detect_vegetables(image_path)
        st.session_state.detected_items.update(detected)

    if detected:
        st.success(f"Detected: {', '.join(detected)}")
    else:
        st.warning("No recognizable fruits or vegetables detected.")

if st.session_state.detected_items:
    st.markdown("### 🧾 Current Ingredient List:")
    st.write(", ".join(st.session_state.detected_items))

    if st.button("🍽️ Get Recipes"):
        with st.spinner("Calling Pakistani chef LLM..."):
            prompt = build_recipe_prompt(list(st.session_state.detected_items))
            recipes = call_mistral_api(prompt)
            st.markdown("## 🍲 Suggested Recipes")
            st.write(recipes)
else:
    st.info("Upload at least one image to begin.")
