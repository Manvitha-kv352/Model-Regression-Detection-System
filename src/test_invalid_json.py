
import json

raw_output = "The category is billing."

try:
    parsed_output = json.loads(raw_output)
    error = None

except json.JSONDecodeError as exc:
    parsed_output = None
    error = f"Invalid JSON: {exc}"

print("Raw output:", raw_output)
print("Parsed output:", parsed_output)
print("Error:", error)
