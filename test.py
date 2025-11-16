from core import AudioSearcher

# ==========================================
#  OPTIONAL: PRETTY PRINT RESULTS
# ==========================================
def print_audio_results(results):
    print("\n================= AUDIO SEARCH RESULTS =================\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['filename']}  ({r['folder']})")
        print(f"     Similarity: {r.get('similarity', 'N/A'):.4f}" if 'similarity' in r else "")
        print(f"     Emotion   : {r.get('emotion', None)}")
        print(f"     Transcript: {r['transcript'][:120]}...")
        print(f"     Path      : {r['full_path']}")
        print()
    print("========================================================\n")


searcher = AudioSearcher()

# Πρώτη φορά: θα χτίσει transcripts + embeddings
results = searcher.search_semantic_emotion("καλησπέρα", top_k=5)
print_audio_results(results)
