
import os

filepath = r"d:\Internship\Orbimesh\backend\agents\mail_agent\agent.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Goal: Replace the complex ROBUST ATTACHMENT RESOLUTION block with simple logic
start_idx = -1
for i, line in enumerate(lines):
    if '# ROBUST ATTACHMENT RESOLUTION' in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find start")
    exit(1)

# Find end of loop
end_idx = -1
for i in range(start_idx, len(lines)):
    if 'req = SendEmailRequest(' in lines[i]:
        end_idx = i
        break

if end_idx == -1:
    print("Could not find end")
    exit(1)

indent = lines[start_idx].find('#')

new_logic = [
    " " * indent + "# SIMPLIFIED ATTACHMENT RESOLUTION: If no IDs, take all from history\n",
    " " * indent + "if not attachment_file_ids:\n",
    " " * indent + "    logger.info(f\"DEBUG: No IDs provided. Scanning history (size={len(results)})...\")\n",
    " " * indent + "    for r in results:\n",
    " " * indent + "        res_data = r.get(\"result\", {})\n",
    " " * indent + "        if isinstance(res_data, dict) and \"files\" in res_data:\n",
    " " * indent + "            files = res_data[\"files\"]\n",
    " " * indent + "            found_ids = [f.get(\"file_id\") for f in files if isinstance(f, dict) and f.get(\"file_id\")]\n",
    " " * indent + "            logger.info(f\"DEBUG: Found {len(found_ids)} files in step {r.get('step')}: {found_ids}\")\n",
    " " * indent + "            attachment_file_ids.extend(found_ids)\n",
    "\n"
]

final_lines = lines[:start_idx] + new_logic + lines[end_idx:]

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print("Successfully applied simplified logic")
