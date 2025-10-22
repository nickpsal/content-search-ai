from core import ImageSearcher

search_query = "man working in the bathroom"
DATA_DIR = "./data"

searcher = ImageSearcher(data_dir=DATA_DIR)
searcher.download_coco_data()
searcher.extract_image_embeddings(folder="val2017")
searcher.extract_image_embeddings(folder="other")
searcher.extract_text_embeddings()
searcher.search(search_query, top_k=5)
