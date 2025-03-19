from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
import chromadb
import time


class faiss_db:
    def __init__(self, model=None):
        if model is None:
            self.embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
        else:
            self.embedding_model = model
        
        self.retriever=None
    
    def get_embedding(self, docs):
        """Generates an embedding for the given text."""
        vectordb=FAISS.from_documents(docs,self.embedding_model)
        print(f"Size of vector DB : {vectordb.index.ntotal}")
        self.retriever=vectordb.as_retriever()

class chroma_db:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
        self.requirements_collection = self.chroma_client.get_or_create_collection(name="requirements")
        time.sleep(1)
        self.us_collection = self.chroma_client.get_or_create_collection(name="userstories")
    
    def add_requirements(self, requirements):
        req_no = 1
        print(f"i have got the requirement as : {requirements}, type: {type(requirements)}")
        if isinstance(requirements, str):
            requirements = [requirements]  # Convert single string to list
            

        for requirement in requirements:
            if requirement:
                req_text = f"Requirement: {requirement}"
                id=f"REQ-{req_no}"
                self.upsert_requirement( id, req_text)
                req_no += 1
            
    def get_existing_requirement(self, req_id):
        """Retrieve requirement by ID"""
        results = self.requirements_collection.get(ids=[req_id])
        return results if results["documents"] else None
    
    def upsert_requirement(self, req_id, new_text):
        """Compare new requirement with existing and update or add accordingly, auto-assigning status."""
        
        existing_req = self.get_existing_requirement(req_id)

        if existing_req:
            existing_text = existing_req["documents"][0]
            existing_metadata = existing_req["metadatas"][0]

            # Check if requirement text has changed
            if existing_text != new_text:
                print(f"Updating requirement: {req_id}")

                # Retain previous status or mark as "updated"
                updated_metadata = existing_metadata
                updated_metadata["status"] = updated_metadata.get("status", "draft")  # Default to draft if missing
                updated_metadata["status"] = "updated"

                # Delete old version before inserting updated one
                self.requirements_collection.delete(ids=[req_id])
                self.requirements_collection.add(ids=[req_id], documents=[new_text], metadatas=[updated_metadata])
            else:
                print(f"No changes detected for: {req_id}, skipping update.")
        else:
            print(f"Adding new requirement: {req_id}")

            # Assign default metadata with "new" status
            new_metadata = {"phase": "requirements", "status": "new"}
            self.requirements_collection.add(ids=[req_id], documents=[new_text], metadatas=[new_metadata])

    ## for User stories
    def add_ustories(self, us):
            req_no = 1
            ustories=us
            print(f"i have got the user stories as : {us}, type: {type(us)}")
            if isinstance(us, str):
                ustories = [us]  # Convert single string to list

            for ustory in ustories:
                if ustory:
                    req_text = f"User Story: {ustory}"
                    id=f"US-{req_no}"
                    self.upsert_us(id, req_text)
                    req_no += 1
            
    def get_existing_ustory(self, us_id):
        """Retrieve User Story by ID"""
        results = self.us_collection.get(ids=[us_id])
        return results if results["documents"] else None
    
    def upsert_us(self, req_id, new_text):
        """Compare new User story with existing and update or add accordingly, auto-assigning status."""
        
        existing_req = self.get_existing_ustory(req_id)

        if existing_req:
            existing_text = existing_req["documents"][0]
            existing_metadata = existing_req["metadatas"][0]

            # Check if requirement text has changed
            if existing_text != new_text:
                print(f"Updating User story: {req_id}")

                # Retain previous status or mark as "updated"
                updated_metadata = existing_metadata
                updated_metadata["status"] = updated_metadata.get("status", "draft")  # Default to draft if missing
                updated_metadata["status"] = "updated"

                # Delete old version before inserting updated one
                self.us_collection.delete(ids=[req_id])
                self.us_collection.add(ids=[req_id], documents=[new_text], metadatas=[updated_metadata])
            else:
                print(f"No changes detected for: {req_id}, skipping update.")
        else:
            print(f"Adding new User Story: {req_id}")

            # Assign default metadata with "new" status
            new_metadata = {"phase": "User Story", "status": "new"}
            self.us_collection.add(ids=[req_id], documents=[new_text], metadatas=[new_metadata])
