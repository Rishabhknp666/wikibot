from sqlalchemy import null
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain 
import os
import wikipediaapi 
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup






with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
     -[streamlit](https://streamlit.io/)
    -[Langchain](https://www.langchain.com/)
    -[OpenAI](https://openai.com/)
                

    ''')

    add_vertical_space(5)
    st.write('Made with love by [Rishabh](https://youtube.com/playlist?list=PLdMtv-iP-2mQh_6HUSQmGxn6ByOAYd4bY&si=iiyDaltdXERhOHGv )')


def getwsummary(usrinpt):
   wiki = wikipediaapi.Wikipedia('wikibot.en')
   page = wiki.page(usrinpt)


   if not page.exists():
       return "Invalid wikipedia link. Enter a valid one"
   else:
        def get_wiki_text(url):
    
            response = requests.get(usrinpt)

    
            if response.status_code == 200:
        
                soup = BeautifulSoup(response.content, 'html.parser')

        
                main_content = soup.find('div', class_='mw-content-text')

        
                paragraphs = main_content.find_all('p') # type: ignore
                extracted_text1 = "\n".join([paragraph.get_text() for paragraph in paragraphs])

                return extracted_text1
    



def main():
    st.header("chat with wiki")

    usrinpt = st.text_input("Enter your wikipedia url"," ")
    query = st.text_input("You:"," ")


    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=300,
        length_function=len 
    )
    extracted_text = getwsummary(usrinpt)
    chunks = text_splitter.split_text(text=extracted_text) # type: ignore

    parsed_url = urlparse(usrinpt)
    store_name = parsed_url.path[:-1] if parsed_url.path.endswith('/') else parsed_url.path
    embeddings=None

    if os.path.exists(f"{store_name}.pkl"  ):
        with open(f"{store_name}.pkl","rb") as f:
             VectorStore = pickle.load(f)
      #st.write("embeddings loaded from the disk")
    else:
        st.write("upload a wiki file")     


    embeddings = OpenAIEmbeddings() # type: ignore
        
    VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
    with open(f"{store_name}.pkl","wt" ) as f:
         pickle.dump(VectorStore,f) # type: ignore

    st.write("embeddings computation completed")

         #accept user question query
    query = st.text_input("ask questions about your wiki page")
    st.write(query)

    if query is not null: 
        VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        docs = VectorStore.similarity_search(query=query,k=3)

        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm=llm,chain_type="stuff")
        response = chain.run(input_documents=docs,question=query)
        st.write(response)

        






    if st.button("send"):
        response = OpenAI.Completion.create(
            engiine="chatgpt 3.5 turbo",
            prompt=usrinpt,
            max_token=100
        )

        st.text("Chatbot:" + response.choices[0].text)
    






if __name__ == "__main__":
    main()
