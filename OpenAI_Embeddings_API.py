import os 
os.environ["OPENAI_API_KEY"] = "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"


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


llm = OpenAI()
print(llm("tell me a joke"))

from langchain.chains.question_answering import load_qa_chain



loader = PyPDFLoader("materials/example.pdf")
documents = loader.load()
# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
query = "what is the total number of AI publications?"
result = qa({"query": query})

retriever.get_relevant_documents(query)

desired_keys = ["query", "result"]

for key,value in result.items():
    if key in desired_keys:
        print(f"{key}: {value}")