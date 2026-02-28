import os
import re

directories = ['app', 'components', 'lib', 'workers']
for d in directories:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(('.ts', '.tsx')):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # A simple regex to remove single-line console.logs
                # Be careful not to remove multiline cleanly if not easy, but usually they are 1-liners
                new_content = re.sub(r'^\s*console\.log\([^)]*\);?\s*?\n', '', content, flags=re.MULTILINE)
                
                # Also remove console.log that might be a bit more complex (e.g. nested brackets)
                # But safer to just use a slightly broader regex for typical logs
                new_content = re.sub(r'^\s*//\s*console\.log.*\n', '', new_content, flags=re.MULTILINE)

                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Cleaned {filepath}")
