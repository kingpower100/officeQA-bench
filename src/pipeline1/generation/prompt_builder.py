PROMPT_TEMPLATE_VERSION = "v2_strict_final_answer"


def build_prompt(system_prompt: str, question: str, contexts: list) -> str:
    context_text = "\n\n".join(f"[{idx}] {item.text}" for idx, item in enumerate(contexts, start=1))
    return (
        f"{system_prompt.strip()}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        "Final Answer:"
    )


def dedupe_prompt_contexts(contexts: list) -> list:
    seen = set()
    output = []
    for item in contexts:
        key = " ".join(item.text.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output
