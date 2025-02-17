from config import METADATA_PATH
import json
from datetime import datetime
from utils.screenshot_util import load_screenshot_from_path

def get_user_timeline():
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        timeline = []
        for entry in metadata:
            raw_data = entry.get('raw', {})
            
            timestamp = raw_data.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            timeline_entry = {
                'id': entry.get('id'),
                'title': raw_data.get('title', 'Untitled'),
                'url': raw_data.get('url', ''),
                'timestamp': formatted_timestamp,
                'screenshot': load_screenshot_from_path(entry.get('screenshot_path'))
            }
            
            timeline.append(timeline_entry)
        
        timeline.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        
        return timeline
    
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading timeline: {str(e)}")
        return []
    

if __name__ == "__main__":
    timeline = get_user_timeline()
    print(timeline[0].keys())