from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeLangchain
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["PINECONE_API_KEY"] = "d7997333-805a-433c-82f2-77e3010afb6a"

api_key = os.environ.get("PINECONE_API_KEY")
print("Pinecone API Key:", api_key) 

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("/Users/aakhilabakwad/PycharmProjects/End-to-end-Medical-Chatbot-using-Llama2/data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

api_key = "d7997333-805a-433c-82f2-77e3010afb6a"

pc = Pinecone(api_key=api_key)

if 'medical-chatbot' not in pc.list_indexes().names():
    pc.create_index(
        name='medical-chatbot',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

print("Index setup complete.")

docsearch=PineconeLangchain.from_texts([t.page_content for t in text_chunks], embeddings, index_name="medical-chatbot")
