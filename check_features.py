"""
Script kiểm tra features trong database
"""
import pickle
import os

db_path = 'image_database.pkl'

if os.path.exists(db_path):
    with open(db_path, 'rb') as f:
        images = pickle.load(f)
    
    print(f"Total images: {len(images)}\n")
    
    # Kiểm tra vài ảnh đầu
    for idx, (img_id, img_data) in enumerate(list(images.items())[:3], 1):
        print(f"[{idx}] {img_id}")
        
        if img_data['features']:
            features = img_data['features']
            print(f"  Features keys: {list(features.keys())}")
            
            for key in features:
                if key != 'combined':
                    print(f"    {key}: {len(features[key])} dims")
            
            # Kiểm tra có deep không
            if 'deep' in features:
                has_values = features['deep'].any() if hasattr(features['deep'], 'any') else bool(features['deep'])
                print(f"  ✅ HAS DEEP FEATURES: {has_values}")
            else:
                print(f"  ❌ MISSING DEEP FEATURES!")
        else:
            print(f"  No features extracted yet")
        print()
else:
    print("Database file not found!")
