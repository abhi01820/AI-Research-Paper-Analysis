import os,json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.load_and_extract_text import extract_text_from_pdf,extract_pdf_sections
from src.detect_and_split_sections import refine_sections,split_sections_with_content

from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
embedding_model=os.getenv("EMBEDDING_MODEL")
llm_model=os.getenv("LLM_MODEL")

llm=ChatGroq(groq_api_key=groq_api_key,model_name=llm_model)
# print(llm.invoke("who is Hardik Pandya?"))

if __name__=="__main__":
    extracted_text=extract_text_from_pdf("research-paper.pdf")
    extracted_sections=extract_pdf_sections(full_text=extracted_text)
    refined_sections=refine_sections(extracted_sections,llm)
    section_with_content=split_sections_with_content(extracted_text,refined_sections)
    with open("section_with_content.json","w") as f:
        json.dump(section_with_content,f,indent=4)
