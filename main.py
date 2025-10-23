from core import ImageSearcher
from core import Model

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
