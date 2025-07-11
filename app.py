import os
from enhancer import enhance_face

def main():
    input_folder = 'inputs'
    output_folder = 'results'

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"restored_{filename}")
            print(f"[~] Processing: {filename}")
            enhance_face(input_path, output_path)

if __name__ == '__main__':
    main()
