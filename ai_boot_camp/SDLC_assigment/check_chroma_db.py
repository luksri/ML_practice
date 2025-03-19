import chromadb

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# List collections (returns list of collection names as strings)
collection_names = chroma_client.list_collections()

print("Available Collections:", collection_names)

# If collections exist, fetch the first one
if collection_names:
    collection_name = collection_names[0]  # Directly use the first name (string)
    collection = chroma_client.get_collection(name=collection_name)

    # Retrieve stored documents
    data = collection.get()

    print("Documents:", data["documents"])
    print("Metadata:", data["metadatas"])
    print("IDs:", data["ids"])
else:
    print("No collections found in ChromaDB.")
