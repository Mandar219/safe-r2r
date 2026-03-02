import json

PATH = "data/processed/hotpot_qa_distractor_validation_queries.jsonl"

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        first = json.loads(next(f))
    print("qid:", first["qid"])
    print("question:", first["question"][:120])
    print("answer:", first["answer"])
    print("num_context_paras:", len(first["context"]))
    print("first_title:", first["context"][0]["title"])
    print("first_text_snippet:", first["context"][0]["text"][:150])

if __name__ == "__main__":
    main()