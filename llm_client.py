import subprocess
import re

MODEL = 'mistral'

def run_ollama(prompt, model=MODEL, timeout=120):
    cmd = ['ollama', 'run', model]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("ollama timed out while generating a response")
    if process.returncode != 0:
        raise RuntimeError(stderr.strip() or f"ollama exited with code {process.returncode}")
    return clean_terminal_output(stdout).strip()


def clean_terminal_output(text):
    text = text or ""

    # Ollama sometimes emits terminal cursor controls while streaming. Recreate the common "move left, clear, rewrite" behavior before display.
    cursor_clear = re.compile(r"\x1b\[(\d*)D\x1b\[K")
    while True:
        match = cursor_clear.search(text)
        if not match:
            break

        chars_to_remove = int(match.group(1) or 1)
        start = max(0, match.start() - chars_to_remove)
        text = text[:start] + text[match.end():]

    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text
