from pathlib import Path

Path("/content/scripts").mkdir(parents=True, exist_ok=True)
print("cwd", Path.cwd())
print("scripts_dir", Path("/content/scripts"), Path("/content/scripts").exists())
