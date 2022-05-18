import subprocess
import os

try:
    with open(os.devnull, 'wb') as devnull:
        subprocess.call(['git', 'pull'], stdout=devnull, stderr=devnull)
except:
    pass
