from config import load_prompt_config

config = load_prompt_config("prompts/v1.yaml")

print(config.version)
print(config.model)
print(config.temperature)
print(config.system_prompt)