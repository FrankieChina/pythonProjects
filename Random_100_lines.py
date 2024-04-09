import random

def select_random_lines(file_path, num_lines=100, output_file="selected_lines.txt"):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            random_lines = random.sample(lines, num_lines)
            with open(output_file, 'w') as output:
                output.writelines(random_lines)
            return output_file
    except FileNotFoundError:
        return None

# Example usage: replace "your_file.txt" with the actual file path
file_path = "analysis.txt"
output_file = select_random_lines(file_path)

if output_file:
    print(f"Selected lines written to {output_file}")
else:
    print(f"File '{file_path}' not found.")
