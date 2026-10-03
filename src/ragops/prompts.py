PROMPTS = {
    "grounded-v1": (
        "Answer only from the context. If the context is insufficient, say you do not know. "
        "Cite sources as [doc_id].\n\nQuestion: {question}\n\nContext:\n{context}\n"
    ),
    "short-v1": "Q: {question}\nContext:\n{context}\nA:",
}


def render(prompt_version: str, question: str, context: str) -> str:
    template = PROMPTS.get(prompt_version)
    if template is None:
        raise KeyError(f"unknown prompt version {prompt_version}")
    return template.format(question=question, context=context)
