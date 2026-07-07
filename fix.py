import os, re

d = r'c:\Users\kenta\pythonPlayGround'
for f in os.listdir(d):
    if f.endswith('.py') and f not in ['sazae_scraper.py', 'update_logic.py', 'fix.py']:
        p = os.path.join(d, f)
        with open(p, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace output_path assignment completely
        content = re.sub(
            r"output_path\s*=\s*[rR]?['\"][^\n]+['\"]",
            f"output_path = r'c:\\Users\\kenta\\pythonPlayGround\\{f.replace('.py', '.png')}'",
            content
        )
        
        with open(p, 'w', encoding='utf-8') as file:
            file.write(content)

print('Fixed output paths')
