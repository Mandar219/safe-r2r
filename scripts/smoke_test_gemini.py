from src.utils.io import load_yaml
from src.generation.gemini_client import GeminiClient, GeminiConfig

def main():
    cfg = load_yaml("configs/default.yaml")
    g = cfg["gemini"]
    client = GeminiClient(GeminiConfig(**g))
    out = client.generate_text("Say 'Gemini is working' and nothing else.")
    print(out)

if __name__ == "__main__":
    main()