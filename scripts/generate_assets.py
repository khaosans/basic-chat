import os
from PIL import Image

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_assets(logo_path):
    # Create directories
    dirs = [
        'assets/brand/logo',
        'assets/brand/favicon',
        'assets/brand/social'
    ]
    for dir_path in dirs:
        ensure_dir(dir_path)
    
    # Load the logo
    logo = Image.open(logo_path)
    
    # Save full logo
    logo.save('assets/brand/logo/elron-logo-full.png', 'PNG')
    
    # Generate favicons
    favicon_sizes = {
        16: 'favicon-16x16.png',
        32: 'favicon-32x32.png',
        180: 'apple-touch-icon.png'
    }
    
    for size, filename in favicon_sizes.items():
        resized = logo.resize((size, size), Image.LANCZOS)
        resized.save(f'assets/brand/favicon/{filename}')
    
    # Generate social media images
    social_sizes = {
        'og-image.png': (1200, 630),
        'twitter-card.png': (1200, 600)
    }
    
    for filename, (width, height) in social_sizes.items():
        # Create social media image with logo
        social_img = Image.new('RGB', (width, height), (15, 23, 42))  # Dark background
        
        # Calculate dimensions for the logo (use 1/3 of the height)
        logo_height = height // 3
        aspect_ratio = logo.width / logo.height
        logo_width = int(logo_height * aspect_ratio)
        
        # Resize logo
        resized_logo = logo.resize((logo_width, logo_height), Image.LANCZOS)
        
        # Calculate position to center the logo
        x = (width - logo_width) // 2
        y = (height - logo_height) // 2
        
        # Paste the logo
        social_img.paste(resized_logo, (x, y))
        social_img.save(f'assets/brand/social/{filename}')

if __name__ == "__main__":
    generate_assets('LOGO.jpg')
