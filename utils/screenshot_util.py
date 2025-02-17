import os
import base64

def load_screenshot_from_path(screenshot_path: str) -> str | None:
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            with open(screenshot_path, 'rb') as img_file:
                screenshot_data = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{screenshot_data}"
        except Exception as e:
            print(f"Error loading screenshot {screenshot_path}: {str(e)}")
    return None