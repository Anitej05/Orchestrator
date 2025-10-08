import os
import json
import re

CONVERSATION_DIR = 'conversation_history'

def parse_legacy_python_repr(text: str):
    messages = []
    # Try to find messages=[ ... ] or 'messages': [ ... ] block
    m = re.search(r"messages\s*[:=]\s*\[([\s\S]*?)\]\s*[,}]", text)
    block = m.group(1) if m else text

    msg_regex = re.compile(r"(AIMessage|HumanMessage|ChatMessage|SystemMessage)\s*\(([^)]*)\)")
    for mm in msg_regex.finditer(block):
        kind = mm.group(1)
        body = mm.group(2)
        content_match = re.search(r"content\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", body)
        id_match = re.search(r"id\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", body)
        content = (content_match.group(1) or content_match.group(2)) if content_match else ''
        idv = (id_match.group(1) or id_match.group(2)) if id_match else None
        messages.append({
            'id': idv or f'legacy-{len(messages)}',
            'type': 'user' if 'HumanMessage' in kind else 'assistant',
            'content': content,
            'timestamp': None,
            'metadata': {}
        })

    # If we found nothing, try to extract content='...' occurrences
    if not messages:
        for m2 in re.finditer(r"content\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", text):
            content = m2.group(1) or m2.group(2) or ''
            messages.append({
                'id': f'legacy-{len(messages)}',
                'type': 'assistant',
                'content': content,
                'timestamp': None,
                'metadata': {}
            })

    return messages


def fix_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        # Already valid JSON, nothing to do
        return False
    except Exception:
        # Try to parse legacy repr
        messages = parse_legacy_python_repr(raw)
        if not messages:
            # As a fallback, write raw into a structure
            payload = {'raw_state': raw}
        else:
            payload = messages

        # Overwrite file with JSON payload
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True


def main():
    if not os.path.isdir(CONVERSATION_DIR):
        print('No conversation directory found')
        return
    files = [f for f in os.listdir(CONVERSATION_DIR) if f.endswith('.json')]
    fixed = []
    for fn in files:
        path = os.path.join(CONVERSATION_DIR, fn)
        try:
            if fix_file(path):
                fixed.append(fn)
        except Exception as e:
            print('Failed to fix', fn, e)
    print('Fixed files:', fixed)

if __name__ == '__main__':
    main()
