import subprocess

def run_ollama(prompt, model):
    cmd = ['ollama', 'run', model]
    process = subprocess.Popen(cmd, stdin= subprocess.PIPE, stdout= subprocess.PIPE, text=True)
    stdout, _ = process.communicate(prompt)
    return stdout.strip()