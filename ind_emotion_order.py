import torch

# Load model
state = torch.load("models/best_model_audio_emotion_v4.pt", map_location="cpu")

# Detect classifier layer name
classifier_weight = None
classifier_bias = None

for k in state.keys():
    if "weight" in k.lower() and ("classifier" in k.lower() or "fc" in k.lower() or "out" in k.lower()):
        classifier_weight = state[k]
        classifier_bias = state[k.replace("weight", "bias")]
        break

if classifier_weight is None:
    raise Exception("Classifier layer not found.")

# Fake embedding
fake = torch.randn(1, classifier_weight.shape[1])

# Compute logits
logits = fake @ classifier_weight.T + classifier_bias
logits = logits.flatten().tolist()

print("Logit values:", logits)
print("Sorted index order (from highest to lowest):")
print(sorted(list(range(len(logits))), key=lambda i: logits[i], reverse=True))
