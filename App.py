import streamlit as st
from transformers import pipeline

st.title("🤖 AI Text Generator")
st.write("Enter some text and click the Generate button.")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M"
    )

generator = load_model()

user_input = st.text_input("Enter your text:")

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating... Please wait"):
            result = generator(
                user_input,
                max_new_tokens=50,
                num_return_sequences=2,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )

        for i, output in enumerate(result, 1):
            st.subheader(f"Output {i}")
            st.write(output["generated_text"])
    else:
        st.warning("Please enter some text!")
