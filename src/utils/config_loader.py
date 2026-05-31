import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_data_schema(schema_provided_path: Optional[str] = None) -> Dict[str, Any]:    
    """
    Load schema file. 
    - Nếu có schema_provided_path: Ưu tiên dùng đường dẫn này (thường là từ MLflow context).
    - Nếu không: Tìm kiếm theo các đường dẫn fallback (cho local dev).
    """

    potential_paths = []
    if schema_provided_path:
        path_obj = Path(schema_provided_path)
        if path_obj.exists():
            potential_paths.append(path_obj)
        else:
            print(f"Warning: Đường dẫn được cung cấp không tồn tại: {schema_provided_path}")

    root_dir = Path(__file__).resolve().parent.parent   # Bổ sung đường dẫn Fallback (Nhớ đẩy nguyên thư mục src/ hoặc src/config/ lên nơi chạy code)
    default_schema_rel = "config/schema.yaml"
    potential_paths.extend([
        Path.cwd() / default_schema_rel,
        root_dir / default_schema_rel,
    ])
    
    # 2. Tìm đường dẫn tồn tại
    print(f"Debug: Đang tìm file config schema của data tại {potential_paths}")
    full_path = next((p for p in potential_paths if p.exists()), None)
    if not full_path:
        raise FileNotFoundError(f"Không tìm thấy file schema. Đã thử tìm tại các vị trí: {potential_paths}")
            
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"File YAML tại {full_path} không hợp lệ: {e}")
    except Exception as e:
        raise IOError(f"Không thể đọc file tại {full_path}: {e}")