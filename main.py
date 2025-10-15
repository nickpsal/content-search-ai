from core import ImageSearcher

search_query = "women on beach"
searcher = ImageSearcher()
searcher.download_coco_data()
searcher.extract_image_embeddings()
searcher.extract_text_embeddings()
searcher.search(search_query)