
import os

filepath = r"d:\Internship\Orbimesh\backend\agents\mail_agent\agent.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Target block (carefully constructed from what I saw)
# We use a broad search string to find the start
search_start = 'elif "send" in step_action:'
pos = content.find(search_start)
if pos == -1:
    print("Could not find start block")
    exit(1)

# Find the end of that block (next elif or else or end of function)
# The block ends around 'if send_res["success"]: result = send_res["data"]'
# Actually, let's just replace the whole section between 'elif "send" in step_action:' and 'else:' (unknown action)

target_end = 'else:'
end_pos = content.find(target_end, pos)
if end_pos == -1:
    print("Could not find end block")
    exit(1)

new_block = """elif "send" in step_action:
                            # Handling sending email with attachments
                            # Params: to, subject, body, attachment_paths using keys from previous results
                            
                            # 1. Handle RECIPIENT ('to') - Ensure it's a list
                            to_param = step_params.get("to", ["me"])
                            if isinstance(to_param, str):
                                to_list = [to_param]
                            elif isinstance(to_param, list):
                                to_list = to_param
                            else:
                                to_list = ["me"]
                                
                            # 2. Handle ATTACHMENTS
                            attachment_file_ids = step_params.get("attachment_file_ids", [])
                            attachment_paths = step_params.get("attachment_paths", [])
                            
                            # Check if we should use history/history of this task
                            should_use_history = step_params.get("use_history", False)
                            
                            if not attachment_file_ids and should_use_history:
                                # Try to find downloaded files from previous steps in this task
                                for r in results:
                                    if "download" in r.get("step", ""):
                                        res_data = r.get("result", {})
                                        # download_email_attachments returns {'success': True, 'files': [...]}
                                        if isinstance(res_data, dict) and "files" in res_data:
                                            # format: [{'file_id':..., 'file_path':...}]
                                            attachment_file_ids.extend([f['file_id'] for f in res_data['files']])

                            req = SendEmailRequest(
                                to=to_list,
                                subject=step_params.get("subject", "Automated Reply"),
                                body=step_params.get("body", "Sent via Mail Agent"),
                                attachment_file_ids=attachment_file_ids if isinstance(attachment_file_ids, list) else [],
                                attachment_paths=attachment_paths if isinstance(attachment_paths, list) else [],
                                user_id=step_params.get("user_id", "me")
                            )
                            
                            try:
                                send_res = await gmail_client.send_email_with_attachments(
                                    to=req.to,
                                    subject=req.subject,
                                    body=req.body,
                                    attachment_file_ids=req.attachment_file_ids,
                                    attachment_paths=req.attachment_paths,
                                    user_id=req.user_id
                                )
                                if send_res["success"]: result = send_res["data"]
                                else: execution_error = send_res.get("error")
                            except Exception as e:
                                execution_error = str(e)

                        """

# Note: We need to preserve the indentation of the original 'elif'
# elif "send" in step_action: is indented by some amount.
# Let's see how much.
line_start = content.rfind("\n", 0, pos) + 1
indent = pos - line_start
print(f"Detected indent: {indent}")

# Re-indent the new block
lines = new_block.splitlines()
# First line already has some spaces in my string, let's fix it.
first_line = lines[0].strip()
other_lines = [lines[i] for i in range(1, len(lines))]
# Find minimal indent in other lines to strip and re-indent
min_other_indent = 100
for l in other_lines:
    if l.strip():
        min_other_indent = min(min_other_indent, len(l) - len(l.lstrip()))

reindented_lines = [" " * indent + first_line]
for l in other_lines:
    if l.strip():
        reindented_lines.append(" " * indent + l[min_other_indent:])
    else:
        reindented_lines.append("")

final_new_block = "\n".join(reindented_lines) + "\n\n"

# Replace
updated_content = content[:pos] + final_new_block + content[end_pos:]

with open(filepath, "w", encoding="utf-8") as f:
    f.write(updated_content)

print("Successfully patched agent.py")
