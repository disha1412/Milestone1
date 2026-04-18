from PIL import Image
import numpy as np
rng = np.random.default_rng(0)
for name in ["test_a.jpg", "test_b.jpg"]:
    img = Image.fromarray(rng.integers(0, 256, (160, 160, 3), dtype=np.uint8))
    img.save(name)
print("Saved test_a.jpg and test_b.jpg")