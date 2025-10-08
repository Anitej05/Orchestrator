import json, re, os
root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'conversation_history'))
if not os.path.isdir(root):
    print('No repo-root conversation_history dir at', root)
    raise SystemExit(0)
files=[f for f in os.listdir(root) if f.endswith('.json')]
changed=[]
for fn in files:
    path=os.path.join(root,fn)
    s=open(path,'r',encoding='utf-8').read()
    try:
        json.loads(s)
        # already json
        continue
    except Exception:
        pass
    m=re.search(r"messages\s*[:=]\s*\[([\s\S]*?)\]\s*[,}]", s)
    msgs=[]
    if m:
        inner=m.group(1)
        for mm in re.finditer(r"(AIMessage|HumanMessage|ChatMessage|SystemMessage)\s*\(([^)]*)\)", inner):
            kind=mm.group(1); body=mm.group(2)
            content_match=re.search(r"content\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", body)
            id_match=re.search(r"id\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", body)
            content=(content_match.group(1) or content_match.group(2)) if content_match else ''
            idv=(id_match.group(1) or id_match.group(2)) if id_match else None
            msgs.append({'id': idv or f'legacy-{len(msgs)}', 'type': 'user' if 'HumanMessage' in kind else 'assistant', 'content': content, 'timestamp': None, 'metadata': {}})
    if not msgs:
        for m2 in re.finditer(r"content\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", s):
            content=m2.group(1) or m2.group(2) or ''
            msgs.append({'id': f'legacy-{len(msgs)}', 'type':'assistant','content':content,'timestamp':None,'metadata':{}})
    if msgs:
        open(path,'w',encoding='utf-8').write(json.dumps(msgs,ensure_ascii=False,indent=2))
        changed.append(fn)
print('fixed',changed)
