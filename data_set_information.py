import json

# Load your JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract intents
intents = data['intents']

# Basic stats
total_tags = len(intents)
all_tags = [intent['tag'] for intent in intents]
total_patterns = sum(len(intent['patterns']) for intent in intents)
total_responses = sum(len(intent['responses']) for intent in intents)

output = []


# print(f"Total Tags (Intents): {total_tags}")
# print(f"All Tags: {all_tags}")
# print(f"Total Patterns: {total_patterns}")
# print(f"Total Responses: {total_responses}")
# print("\nPatterns and Responses per Tag:")

output.append(f"Total Tags (Intents): {total_tags}")
output.append(f"All Tags: {all_tags}")
output.append(f"Total Patterns: {total_patterns}")
output.append(f"Total Responses: {total_responses}")
output.append("\nPatterns and Responses per Tag:")

# Detailed stats
for intent in intents:
    tag = intent['tag']
    num_patterns = len(intent['patterns'])
    num_responses = len(intent['responses'])
    # print(f" - Tag: {tag}")
    # print(f"    Patterns: {num_patterns}")
    # print(f"    Responses: {num_responses}")
    output.append(f" - Tag: {tag}")
    output.append(f"    Patterns: {num_patterns}")
    output.append(f"    Responses: {num_responses}")

with open('dataset_information.txt', 'w') as f:
    f.write('\n'.join(output))

print("Dataset information has been saved to 'dataset_information.txt'")
