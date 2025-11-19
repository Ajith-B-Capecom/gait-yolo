from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
import numpy as np

# -----------------------------
# Step 1: Connect to Milvus
# -----------------------------
connections.connect(alias="default", host="127.0.0.1", port="19530")
print("Connected to Milvus successfully")

# -----------------------------
# Step 2: Create a Collection
# -----------------------------
dim = 8  # small dimension for learning/testing

fields = [
    FieldSchema(name="gait_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="person_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, description="Simple test embeddings")

# Create collection if not exists
collection_name = "test_gait_collection"
if collection_name not in utility.list_collections():
    collection = Collection(collection_name, schema=schema)
    print(f"Collection '{collection_name}' created successfully")
else:
    collection = Collection(collection_name)
    print(f"Collection '{collection_name}' loaded successfully")

# -----------------------------
# Step 3: Insert Sample Data
# -----------------------------
person_ids = ["person_001", "person_002", "person_003"]
embeddings = [np.random.rand(dim).astype("float32") for _ in person_ids]

collection.insert([person_ids, embeddings])
collection.flush()
print(f"Inserted {collection.num_entities} vectors")

# -----------------------------
# Step 4: Check Stored Data
# -----------------------------
print("Total vectors in collection:", collection.num_entities)
print("Collections in DB:", utility.list_collections())
