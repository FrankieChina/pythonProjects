import random
import sys

def select_random_lines(file_path, num_lines=100, output_file="random_nf.txt"):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            random_lines = random.sample(lines, num_lines)
            with open(output_file, 'w') as output:
                output.writelines(random_lines)
            return output_file
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python random_line_selector.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = select_random_lines(input_file)

    if output_file:
        print(f"Selected lines written to {output_file}")
    else:
        print(f"File '{input_file}' not found.")
