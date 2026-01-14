
import os

filepath = r"d:\Internship\Orbimesh\backend\agents\mail_agent\agent.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Goal: find lines between 'elif "send" in step_action:' and 'else:' (unknown action)
# and ensure they have 28 spaces if the elif has 24.

start_idx = -1
end_idx = -1
indent = 24

for i, line in enumerate(lines):
    if 'elif "send" in step_action:' in line:
        start_idx = i
        indent = line.find('elif')
        break

if start_idx == -1:
    print("Could not find start")
    exit(1)

for i in range(start_idx + 1, len(lines)):
    # Look for the next 'else:' at the same indentation level
    if lines[i].startswith(" " * indent + "else:"):
        end_idx = i
        break

if end_idx == -1:
    print("Could not find end")
    exit(1)

print(f"Found block from line {start_idx+1} to {end_idx+1} with indent {indent}")

# Construct the new block content with correct indentation (indent + 4)
new_lines = [
    lines[start_idx], # keep the elif line
    " " * (indent + 4) + "# Handling sending email with attachments\n",
    " " * (indent + 4) + "# Params: to, subject, body, attachment_paths using keys from previous results\n",
    "\n",
    " " * (indent + 4) + "# 1. Handle RECIPIENT ('to') - Ensure it's a list\n",
    " " * (indent + 4) + "to_param = step_params.get(\"to\", [\"me\"])\n",
    " " * (indent + 4) + "if isinstance(to_param, str):\n",
    " " * (indent + 8) + "to_list = [to_param]\n",
    " " * (indent + 4) + "elif isinstance(to_param, list):\n",
    " " * (indent + 8) + "to_list = to_param\n",
    " " * (indent + 4) + "else:\n",
    " " * (indent + 8) + "to_list = [\"me\"]\n",
    "\n",
    " " * (indent + 4) + "# 2. Handle ATTACHMENTS\n",
    " " * (indent + 4) + "attachment_file_ids = step_params.get(\"attachment_file_ids\", [])\n",
    " " * (indent + 4) + "attachment_paths = step_params.get(\"attachment_paths\", [])\n",
    "\n",
    " " * (indent + 4) + "# Check if we should use history/history of this task\n",
    " " * (indent + 4) + "should_use_history = step_params.get(\"use_history\", False)\n",
    "\n",
    " " * (indent + 4) + "if not attachment_file_ids and should_use_history:\n",
    " " * (indent + 8) + "# Try to find downloaded files from previous steps in this task\n",
    " " * (indent + 8) + "for r in results:\n",
    " " * (indent + 12) + "if \"download\" in r.get(\"step\", \"\"):\n",
    " " * (indent + 16) + "res_data = r.get(\"result\", {})\n",
    " " * (indent + 16) + "# download_email_attachments returns {'success': True, 'files': [...]}\n",
    " " * (indent + 16) + "if isinstance(res_data, dict) and \"files\" in res_data:\n",
    " " * (indent + 20) + "# format: [{'file_id':..., 'file_path':...}]\n",
    " " * (indent + 20) + "attachment_file_ids.extend([f['file_id'] for f in res_data['files']])\n",
    "\n",
    " " * (indent + 4) + "req = SendEmailRequest(\n",
    " " * (indent + 8) + "to=to_list,\n",
    " " * (indent + 8) + "subject=step_params.get(\"subject\", \"Automated Reply\"),\n",
    " " * (indent + 8) + "body=step_params.get(\"body\", \"Sent via Mail Agent\"),\n",
    " " * (indent + 8) + "attachment_file_ids=attachment_file_ids if isinstance(attachment_file_ids, list) else [],\n",
    " " * (indent + 8) + "attachment_paths=attachment_paths if isinstance(attachment_paths, list) else [],\n",
    " " * (indent + 8) + "user_id=step_params.get(\"user_id\", \"me\")\n",
    " " * (indent + 4) + ")\n",
    "\n",
    " " * (indent + 4) + "try:\n",
    " " * (indent + 8) + "send_res = await gmail_client.send_email_with_attachments(\n",
    " " * (indent + 12) + "to=req.to,\n",
    " " * (indent + 12) + "subject=req.subject,\n",
    " " * (indent + 12) + "body=req.body,\n",
    " " * (indent + 12) + "attachment_file_ids=req.attachment_file_ids,\n",
    " " * (indent + 12) + "attachment_paths=req.attachment_paths,\n",
    " " * (indent + 12) + "user_id=req.user_id\n",
    " " * (indent + 8) + ")\n",
    " " * (indent + 8) + "if send_res[\"success\"]: result = send_res[\"data\"]\n",
    " " * (indent + 8) + "else: execution_error = send_res.get(\"error\")\n",
    " " * (indent + 4) + "except Exception as e:\n",
    " " * (indent + 8) + "execution_error = str(e)\n",
    "\n"
]

final_lines = lines[:start_idx] + new_lines + lines[end_idx:]

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print("Successfully applied precise indent patch")
