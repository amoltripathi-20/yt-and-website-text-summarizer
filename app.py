import streamlit as st
import validators

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

# ---------------------------
# URL Input
# ---------------------------
generic_url = st.text_input(
    "URL",
    placeholder="Enter YouTube or Website URL"
)

# ---------------------------
# Prompt Template
# ---------------------------
prompt_template = """
Provide a detailed summary of the following content in approximately 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# ---------------------------
# Button Action
# ---------------------------
if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key.")
        st.stop()

    if not generic_url:
        st.error("Please enter a URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please enter a valid URL.")
        st.stop()

    try:
        with st.spinner("Loading and summarizing content..."):

            # Supported Groq Model
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key,
                temperature=0
            )

            # ---------------------------
            # Load Documents
            # ---------------------------
            if (
                "youtube.com" in generic_url
                or "youtu.be" in generic_url
            ):
                loader = YoutubeLoader.from_youtube_url(
                    generic_url,
                    add_video_info=True
                )
            else:
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

            docs = loader.load()

            # ---------------------------
            # Split Large Documents
            # ---------------------------
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(docs)

            # ---------------------------
            # Summarization Chain
            # ---------------------------
            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce"
            )

            summary = chain.run(docs)

            st.success("Summary Generated Successfully!")
            st.write(summary)

    except Exception as e:
        st.error(f"Error: {str(e)}")
