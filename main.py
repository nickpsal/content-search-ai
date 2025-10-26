from core import ImageSearcher
from core import Model
from core import PDFSearcher

# search_query = "άνδρας με ακουστικά headset και γυαλιά σε γραφείο με υπολογιστές, νυχτερινή βάρδια"
# DATA_DIR = "./data"
#
# model = Model()
# model.download_model()
# searcher = ImageSearcher(data_dir=DATA_DIR)
# searcher.download_coco_data()
# searcher.extract_image_embeddings(folder="val2017")
# searcher.extract_image_embeddings(folder="other")
# searcher.extract_text_embeddings()
# searcher.search(search_query, top_k=5)
#
# query_image = "data/images/query/viber_image_2023-04-18_14-56-20-020.jpg"
# results = searcher.search_by_image(query_image, top_k=5)


searcher = PDFSearcher("./models/mclip_finetuned_coco_ready")

folder = "./data/pdfs"

# results = searcher.search_by_text("deep learning", "./data/pdfs", top_k=3)
#
# for r in results:
#     print(f"\n📄 {r['file']} (page {r['page']}) - score={r['score']:.4f}")
#     print(f"👉 Snippet: {r['snippet']}")


query_pdf = "./data/query/6. Feedforward Neural Networks.pdf"

results = searcher.search_similar_pdfs(query_pdf, folder="./data/pdfs", top_k=5)
for name, score in results:
    print(f"{name} (similarity={score:.4f})")