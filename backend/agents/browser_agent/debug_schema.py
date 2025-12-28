
from .schemas import ActionPlan
import json

bad_json = """
{
  "reasoning": "test",
  "actions": [
    {"name": "click", "xpath": "//tab[@name='Academics']"}
  ]
}
"""

print(f"Attempting to validate: {bad_json}")
try:
    plan = ActionPlan.model_validate_json(bad_json)
    print("Validation success!")
    action = plan.actions[0]
    print(f"Action: {action}")
    print(f"Params keys: {list(action.params.keys())}")
    
    if 'xpath' in action.params:
        print("SUCCESS: xpath found in params")
    else:
        print("FAILURE: xpath NOT found in params")

except Exception as e:
    print(f"Validation failed: {e}")
