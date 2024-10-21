import base64

def decode_base64_to_image(base64_string, output_file_path):
    image_data = base64.b64decode(base64_string)
    with open(output_file_path, 'wb') as output_file:
        output_file.write(image_data)

if __name__ == '__main__':
    base64_string = "your_base64_encoded_string_here"
    output_file_path = "output_image.png"
    decode_base64_to_image(base64_string, output_file_path)