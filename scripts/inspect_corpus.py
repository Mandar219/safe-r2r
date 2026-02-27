import json

CORPUS = "data/processed/hotpot_qa_distractor_corpus.jsonl"
LOOKUP = "data/processed/hotpot_qa_distractor_doc_lookup.json"
CANDS  = "data/processed/hotpot_qa_distractor_validation_query_doc_candidates.jsonl"

def main():
    # corpus sample
    with open(CORPUS, "r", encoding="utf-8") as f:
        first = json.loads(next(f))
    print("First corpus doc:", first["doc_id"], first["title"])
    print("Text snippet:", first["text"][:150])

    # lookup sample
    with open(LOOKUP, "r", encoding="utf-8") as f:
        lookup = json.load(f)
    any_id = next(iter(lookup.keys()))
    print("\nLookup sample doc_id:", any_id)
    print("Lookup title:", lookup[any_id]["title"])

    # candidates check
    with open(CANDS, "r", encoding="utf-8") as f:
        c0 = json.loads(next(f))
    print("\nCandidate row sample qid:", c0["qid"])
    print("Num candidates:", len(c0["candidate_doc_ids"]))
    print("First 3 candidates:", c0["candidate_doc_ids"][:3])

if __name__ == "__main__":
    main()