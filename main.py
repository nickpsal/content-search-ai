from core import ImageSearcher
#from core import Model

search_query = "Ενα άλογο στην παραλία"
DATA_DIR = "./data"
searcher = ImageSearcher(data_dir=DATA_DIR)
searcher.download_coco_data()
searcher.extract_image_embeddings()
searcher.extract_text_embeddings()
searcher.search(search_query)


