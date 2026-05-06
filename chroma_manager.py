import os
import shutil
import streamlit as st
import time

      
def reset_chat_db():
    
    # Clear cached DB instance FIRST
    st.cache_resource.clear()
    
    # give OS time to release file
    time.sleep(0.5)  
 
    if os.path.exists("chroma_chat"):
        shutil.rmtree("chroma_chat")


def reset_handbook_db():
    
    # Clear cached DB instance FIRST
    st.cache_resource.clear()
    
    # give OS time to release file
    time.sleep(0.5)  
    
    if os.path.exists("chroma_handbook"):
        shutil.rmtree("chroma_handbook")
