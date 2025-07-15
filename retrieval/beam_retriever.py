import heapq

class BeamRetriever:
    def __init__(self, beam_width=3, max_steps=2):
        self.beam_width = beam_width
        self.max_steps = max_steps

    def score(self, context, query):
        # Hàm tính điểm đơn giản: số từ trùng nhau
        return len(set(context.lower().split()) & set(query.lower().split()))

    def retrieve(self, contexts, query):
        # contexts: list các đoạn văn bản
        # query: câu hỏi
        beam = []
        for idx, ctx in enumerate(contexts):
            score = self.score(ctx, query)
            heapq.heappush(beam, (-score, idx, [ctx]))
        # Lấy top beam_width kết quả
        top = heapq.nsmallest(self.beam_width, beam)
        results = []
        for neg_score, idx, path in top:
            results.append({
                "score": -neg_score,
                "context": contexts[idx]
            })
        return results

if __name__ == "__main__":
    # Ví dụ sử dụng
    contexts = [
        "Paris là thủ đô của Pháp.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Berlin là thủ đô của Đức."
    ]
    query = "Thủ đô của Việt Nam là gì?"
    retriever = BeamRetriever(beam_width=2)
    results = retriever.retrieve(contexts, query)
    for r in results:
        print(f"Score: {r['score']}, Context: {r['context']}") 