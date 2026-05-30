import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_relative_path: str = "src/config/schema.yaml") -> Dict[str, Any]:
    if os.getenv("DATA_CONFIG_PATH"):                   # Ưu tiên biến môi trường
        potential_paths = [Path(os.getenv("DATA_CONFIG_PATH"))]
    else:
        root_dir = Path(__file__).resolve().parent.parent.parent # : .../project/src/utils/config_loader.py -> .../project
        potential_paths = [
            Path.cwd()  / config_relative_path,         # Thư mục hiện tại (khi chạy từ root: .../project)
            root_dir    / config_relative_path,         # Dùng biến config_relative_path từ root
            root_dir    / "src/config/schema.yaml",     # Đường dẫn tuyệt đối đến file
        ]
    
    # 2. Tìm đường dẫn tồn tại
    full_path = next((p for p in potential_paths if p.exists()), None)
    
    if not full_path:
        raise FileNotFoundError(
            f"Không tìm thấy file config. Đã thử tìm tại các vị trí: {potential_paths}"
        )
            
    with open(full_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)