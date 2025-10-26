from core import ImageSearcher
from core import Model
from core import PDFSearcher

search_query = "άνδρας με ακουστικά headset και γυαλιά σε γραφείο με υπολογιστές, νυχτερινή βάρδια"
DATA_DIR = "./data"

model = Model()
model.download_model()
searcher = ImageSearcher(data_dir=DATA_DIR)
searcher.download_coco_data()
searcher.extract_image_embeddings(folder="val2017")
searcher.extract_image_embeddings(folder="other")
searcher.extract_text_embeddings()
searcher.search(search_query, top_k=5)

query_image = "data/images/query/viber_image_2023-04-18_14-56-20-020.jpg"
results1 = searcher.search_by_image(query_image, top_k=5)
results2 = searcher.search(search_query, top_k=5)

for r in results1:
    name = r.get('name', 'Unknown')
    path = r.get('path', 'N/A')
    score = r.get('score', 0.0)
    print(f"\n🖼️ {name}")
    print(f"📂 Path: {path}")
    print(f"⭐ Similarity: {score:.4f}")
print("=========================================================")
print(f"Search query => {search_query}")
for r in results2:
    name = r.get('name', 'Unknown')
    path = r.get('path', 'N/A')
    score = r.get('score', 0.0)
    print(f"\n🖼️ {name}")
    print(f"📂 Path: {path}")
    print(f"⭐ Similarity: {score:.4f}")