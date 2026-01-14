
import os

filepath = r"d:\Internship\Orbimesh\backend\agents\mail_agent\agent.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Goal: Update the history lookup logic to be more aggressive
start_idx = -1
for i, line in enumerate(lines):
    if 'attachment_file_ids = step_params.get("attachment_file_ids", [])' in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find start")
    exit(1)

# Find the end of the history lookup block
end_idx = -1
for i in range(start_idx, len(lines)):
    if 'req = SendEmailRequest(' in lines[i]:
        end_idx = i
        break

if end_idx == -1:
    print("Could not find end")
    exit(1)

indent = lines[start_idx].find('attachment_file_ids')

new_logic = [
    " " * indent + "attachment_file_ids = step_params.get(\"attachment_file_ids\", [])\n",
    " " * indent + "attachment_paths = step_params.get(\"attachment_paths\", [])\n",
    "\n",
    " " * indent + "# AGGRESSIVE HISTORY LOOKUP: If no attachments provided, always check current task history\n",
    " " * indent + "if not attachment_file_ids and not attachment_paths:\n",
    " " * indent + "    for r in results:\n",
    " " * indent + "        if \"download\" in r.get(\"step\", \"\"):\n",
    " " * indent + "            res_data = r.get(\"result\", {})\n",
    " " * indent + "            if isinstance(res_data, dict) and \"files\" in res_data:\n",
    " " * indent + "                attachment_file_ids.extend([f['file_id'] for f in res_data['files']])\n",
    "\n"
]

final_lines = lines[:start_idx] + new_logic + lines[end_idx:]

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print("Successfully applied aggressive history lookup patch")
