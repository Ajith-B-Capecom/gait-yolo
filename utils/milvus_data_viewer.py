from pymilvus import connections, utility, Collection

# -----------------------------
# Step 1: Connect to Milvus
# -----------------------------
connections.connect(alias="default", host="127.0.0.1", port="19530")
print("Connected to Milvus successfully\n")

# -----------------------------
# Step 2: List all collections
# -----------------------------
collections = utility.list_collections()
print("Collections in DB:", collections, "\n")

# -----------------------------
# Step 3: Inspect each collection
# -----------------------------
for col_name in collections:
    collection = Collection(col_name)
    print(f"--- Collection: {col_name} ---")
    print("Number of vectors:", collection.num_entities)
    print("Fields:", [field.name for field in collection.schema.fields])

    # Determine primary key field for expr
    pk_field = [f.name for f in collection.schema.fields if f.is_primary][0]

    # Preview first 5 entities
    try:
        results = collection.query(
            expr=f"{pk_field} >= 0",  # simple "true" filter
            output_fields=[f.name for f in collection.schema.fields],
            limit=5
        )
        print("\nSample data (first 5 entities):")
        for r in results:
            # Show only first 3 elements of vector for readability
            vec_preview = r['embedding'][:3] if 'embedding' in r else r.get('keypoint_vector', None)
            print({k: (vec_preview if k in ['embedding', 'keypoint_vector'] else v) for k,v in r.items()})
    except Exception as e:
        print("Error fetching data:", e)
    
    print("\n")
