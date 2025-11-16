import torch

PATH = "models/best_model_audio_emotion_v4.pt"

ckpt = torch.load(PATH, map_location="cpu")

print("🔍 Keys found in checkpoint:")
for k in ckpt.keys():
    print(" -", k)
