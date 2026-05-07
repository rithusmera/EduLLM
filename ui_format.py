import re


def normalize_math_text(text):
    text = str(text or "")
    text = normalize_math_delimiters(text)

    text = re.sub(r"\\\[(.*?)\\\]", _block_math, text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", _inline_math, text, flags=re.DOTALL)

    text = re.sub(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", _block_math, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", _block_math, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{aligned\*?\}(.*?)\\end\{aligned\*?\}", _block_math, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}", _block_math, text, flags=re.DOTALL)

    return text


def render_markdown(text, **kwargs):
    import streamlit as st

    text = normalize_math_text(text)
    parts = re.split(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)

    for index, part in enumerate(parts):
        if not part.strip():
            continue

        if index % 2:
            st.latex(part.strip())
        else:
            st.markdown(part, **kwargs)


def normalize_math_delimiters(text):
    return (
        text.replace("\\\\(", "\\(")
        .replace("\\\\)", "\\)")
        .replace("\\\\[", "\\[")
        .replace("\\\\]", "\\]")
    )


def _inline_math(match):
    expr = match.group(1).strip()
    return f"${expr}$"


def _block_math(match):
    expr = match.group(1).strip()
    return f"\n\n$$\n{expr}\n$$\n\n"
