from textwrap import dedent

def synthesize_answer(query, ctxs):
    """
    Basic text-only answer synthesizer for MVP.
    Concatenates retrieved context snippets into a coherent summary.
    """
    # safely format newlines
    bullets = "\n".join(
        ["- " + c["text"][:220].replace("\n", " ") for c in ctxs]
    )

    answer = dedent(f"""
    Question: {query}

    Evidence considered:
    {bullets}

    Synthesis:
    Based on the retrieved evidence, here is a concise answer:
    """).strip()

    # primitive "summary": first sentence of top few docs
    summary = " ".join([c["text"].split(".")[0] for c in ctxs[:3]])
    return answer + "\n" + summary + "."
