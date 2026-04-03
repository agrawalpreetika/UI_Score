from src.clip_scoring import get_clip_score

# 👇 apni image ka path
image_path = "page7.png"

# 👇 category choose karo
category = "landing"   # ya ecommerce / landing / mobile

score = get_clip_score(image_path, category)

print("UI Score:", score)