def format_llm_output(summary, review_text):
    """
    Force consistent bullet formatting for LLM explanations.
    """

    if not summary:
        return summary

    # Remove unwanted headings
    summary = summary.replace("Prediction Analysis based on provided Evidence:", "")
    summary = summary.replace("Prediction Analysis:", "")
    summary = summary.replace("Evidence:", "")

    # Split sentences
    sentences = summary.split(". ")

    bullets = []

    for s in sentences:
        s = s.strip()

        if len(s) < 15:
            continue

        # remove leading markers
        s = s.lstrip("•- ")

        bullets.append(f"• {s}")

    bullets = bullets[:4]  # keep max 4 bullets

    formatted = f'The review "{review_text}"\n\n'
    formatted += "\n\n".join(bullets)

    return formatted