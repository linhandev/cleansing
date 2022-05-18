import subprocess
import os

with open(os.devnull, 'wb') as devnull:
    subprocess.call(['git', 'pull'], stdout=devnull, stderr=devnull)
