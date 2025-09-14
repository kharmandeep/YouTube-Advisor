import weaviate

with weaviate.connect_to_local() as client:
    if client.is_ready():
        print("Successfully connected to Weaviate and the server is ready.")
    else:
        print("Failed to connect to Weaviate or the server is not ready.")
