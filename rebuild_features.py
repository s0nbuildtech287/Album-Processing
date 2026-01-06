"""
Script rebuild tất cả features với deep learning
"""
from services.database import ImageDatabase
from config import Config

print("\n" + "="*70)
print("🔄 FORCE REBUILDING ALL FEATURES WITH DEEP LEARNING")
print("="*70)

# Khởi tạo database
db = ImageDatabase()

images = db.get_all_images()
total = len(images)

print(f"\nTotal images: {total}")
print("Starting feature extraction...\n")

updated = 0
failed = 0

for idx, (image_id, image_data) in enumerate(images.items(), 1):
    try:
        print(f"\n[{idx}/{total}] {image_id}")
        
        # Kiểm tra có deep features chưa
        has_deep = image_data.get('features') and 'deep' in image_data.get('features', {})
        
        if has_deep:
            print(f"  ✓ Already has deep features")
        else:
            print(f"  ⚠️  Missing deep features - extracting...")
        
        # Force extract
        success = db.update_features(image_id)
        
        if success:
            updated += 1
            print(f"  ✅ Successfully updated!")
        else:
            failed += 1
            print(f"  ❌ Failed to update!")
            
    except Exception as e:
        failed += 1
        print(f"  ❌ Error: {e}")

# Lưu database
print("\n" + "="*70)
print("💾 Saving database...")
db.save_database()

print("\n📊 SUMMARY:")
print(f"  Total images: {total}")
print(f"  ✅ Updated: {updated}")
print(f"  ❌ Failed: {failed}")
print("="*70 + "\n")

print("✅ Done! Now you can test query with deep learning features!")
