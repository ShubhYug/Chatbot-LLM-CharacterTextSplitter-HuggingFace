from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def get_qa_chain():
    # Load course content
    loader = TextLoader("codersdaily_courses.txt")
    documents = loader.load()


    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Create vector Chroma-DB using sentence transformer embeddings
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(docs, embedding) #storing chunks in vector db(chroma-DB)
    retriever = vectordb.as_retriever()


    # Load HF LLM (flan-t5)
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)


    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
