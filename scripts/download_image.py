
import requests
import time
from pathlib import Path
from io import BytesIO
from PIL import Image

OUTPUT_DIR = Path("images_2k")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 100
WIDTH = 2048
HEIGHT = 2048

success = 0

for i in range(NUM_IMAGES):
    url = (
        f"https://picsum.photos/seed/image-{i:04d}"
        f"/{WIDTH}/{HEIGHT}"
    )

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))

            if image.size != (WIDTH, HEIGHT):
                raise ValueError("Incorrect image size")

            path = OUTPUT_DIR / f"image_{i+1:03d}.jpg"
            path.write_bytes(image_bytes)

            success += 1
            print(f"[{success}/{NUM_IMAGES}] Saved {path}")
            break

        except Exception as e:
            print(f"Image {i+1}, attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    time.sleep(0.3)

print(f"Downloaded {success}/{NUM_IMAGES} images")
