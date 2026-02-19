#!/usr/bin/env python3
"""
Validate floor plan extraction setup and recommend which method to use.
Checks dependencies, API keys, and system requirements.
"""

import sys
import os
import subprocess
from pathlib import Path
from importlib.util import find_spec

def check_python_version() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    required = (3, 10)

    if version >= required:
        return True, f"Python {version.major}.{version.minor} ✓"
    else:
        return False, f"Python {version.major}.{version.minor} (required 3.10+)"

def check_file(path: str, required: bool = True) -> tuple[bool, str]:
    """Check if file exists."""
    p = Path(path)
    exists = p.exists()

    if exists:
        size = p.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        else:
            size_str = f"{size / 1024:.1f} KB"
        return True, f"{path} ({size_str}) ✓"
    else:
        status = "✗ MISSING" if required else "✗ Not found"
        return exists, f"{path} {status}"

def check_package(package_name: str, import_name: str = None) -> tuple[bool, str]:
    """Check if Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        spec = find_spec(import_name)
        if spec is not None:
            try:
                mod = __import__(import_name)
                version = getattr(mod, "__version__", "unknown")
                return True, f"{package_name} ({version}) ✓"
            except:
                return True, f"{package_name} ✓"
        return False, f"{package_name} not found"
    except (ImportError, ValueError):
        return False, f"{package_name} not found"

def check_env_var(var_name: str) -> tuple[bool, str]:
    """Check if environment variable is set."""
    value = os.getenv(var_name)
    if value:
        # Mask long values
        if len(value) > 30:
            masked = value[:15] + "..." + value[-5:]
        else:
            masked = value
        return True, f"{var_name}={masked} ✓"
    else:
        return False, f"{var_name} not set"

def check_command(cmd: str) -> tuple[bool, str]:
    """Check if command exists."""
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, f"{cmd} available ✓"
    except:
        pass

    return False, f"{cmd} not found"

def main():
    """Run all checks and provide recommendations."""
    print("=" * 100)
    print("FLOOR PLAN EXTRACTION SETUP VALIDATION")
    print("=" * 100)
    print()

    # System checks
    print("SYSTEM REQUIREMENTS")
    print("-" * 100)

    py_ok, py_msg = check_python_version()
    print(f"  {py_msg}")

    if not py_ok:
        print("\n  ERROR: Python 3.10+ required")
        return 1

    print()

    # Core files
    print("CORE FILES")
    print("-" * 100)

    core_files = [
        ("extract_floor_plan.py", True),
        ("tessdata/eng.traineddata", False),
        ("tessdata/osd.traineddata", False),
    ]

    core_ok = True
    for file, required in core_files:
        exists, msg = check_file(file, required)
        status = "✓" if exists else "✗"
        print(f"  {status} {msg}")
        if required and not exists:
            core_ok = False

    print()

    # Python packages
    print("PYTHON PACKAGES")
    print("-" * 100)

    packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pytesseract", "pytesseract"),
    ]

    packages_ok = True
    for pkg, import_name in packages:
        ok, msg = check_package(pkg, import_name)
        status = "✓" if ok else "✗"
        print(f"  {status} {msg}")
        if not ok:
            packages_ok = False

    print()

    # Optional packages for extractors
    print("OPTIONAL PACKAGES (for faster extraction)")
    print("-" * 100)

    optional = [
        ("paddleocr", "paddleocr", "Local fast OCR"),
        ("easyocr", "easyocr", "Alternative OCR"),
        ("google-cloud-vision", "google.cloud.vision", "Google Cloud Vision"),
        ("google-generativeai", "google.generativeai", "Gemini API"),
    ]

    available_methods = ["local"]
    for pkg, import_name, desc in optional:
        ok, msg = check_package(pkg, import_name)
        status = "✓" if ok else "✗"
        print(f"  {status} {msg:<50} ({desc})")

        if "google.cloud.vision" in import_name and ok:
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                available_methods.append("google")
        elif "google.generativeai" in import_name and ok:
            if os.getenv("GEMINI_API_KEY"):
                available_methods.append("gemini")

    print()

    # API keys
    print("API KEYS & CREDENTIALS")
    print("-" * 100)

    gemini_ok, gemini_msg = check_env_var("GEMINI_API_KEY")
    print(f"  {'✓' if gemini_ok else '✗'} {gemini_msg}")

    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if google_creds:
        if Path(google_creds).exists():
            print(f"  ✓ GOOGLE_APPLICATION_CREDENTIALS -> {google_creds}")
            if "google.cloud.vision" in str([pkg for pkg, _, _ in optional]):
                pass  # Already checked above
        else:
            print(f"  ✗ GOOGLE_APPLICATION_CREDENTIALS={google_creds} (file not found)")
    else:
        print(f"  ✗ GOOGLE_APPLICATION_CREDENTIALS not set")

    print()

    # Extraction scripts
    print("EXTRACTION SCRIPTS")
    print("-" * 100)

    scripts = [
        ("extract_floor_plan.py", "Local OCR (PaddleOCR/Tesseract)"),
        ("extract_floor_plan_google_vision.py", "Google Cloud Vision API"),
        ("extract_floor_plan_gemini.py", "Gemini 2.5/2.0 Flash"),
    ]

    for script, desc in scripts:
        exists, msg = check_file(script, False)
        status = "✓" if exists else "✗"
        print(f"  {status} {script:<40} {desc}")

    print()

    # Test images
    print("TEST IMAGES")
    print("-" * 100)

    test_images = [
        "./floor.png",
        "./floor2.png",
        "./floor3.png",
        "./floor4.jpg",
        "./floor5.jpg",
    ]

    found_images = []
    for img in test_images:
        if Path(img).exists():
            found_images.append(img)

    if found_images:
        print(f"  ✓ Found {len(found_images)} test image(s)")
        for img in found_images[:3]:
            exists, msg = check_file(img, False)
            print(f"    • {msg}")
        if len(found_images) > 3:
            print(f"    ... and {len(found_images) - 3} more")
    else:
        print(f"  ✗ No test images found")

    print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    if not core_ok:
        print("✗ Core requirements not met")
        return 1

    print("Available extraction methods:")
    print()

    methods = {
        "local": {
            "name": "Local OCR (PaddleOCR)",
            "speed": "30-40s/image",
            "cost": "Free",
            "setup": "Already available ✓",
            "requirement": "opencv, numpy, pytesseract, paddleocr"
        },
        "google": {
            "name": "Google Cloud Vision",
            "speed": "1-3s/image",
            "cost": "$0.0006/image",
            "setup": "Requires GOOGLE_APPLICATION_CREDENTIALS and google-cloud-vision package",
            "requirement": "API credentials"
        },
        "gemini": {
            "name": "Gemini 2.5 Flash",
            "speed": "<1s/image",
            "cost": "$0.000075/image",
            "setup": "Requires GEMINI_API_KEY and google-generativeai package",
            "requirement": "API key"
        }
    }

    for method in ["local", "google", "gemini"]:
        info = methods[method]
        if method == "local":
            status = "✓ READY"
        else:
            status = "✓ AVAILABLE" if method in available_methods else "✗ NOT AVAILABLE"

        print(f"{status:20} {info['name']}")
        print(f"{'':20} Speed: {info['speed']:15} Cost: {info['cost']:20}")

        if method not in available_methods and method != "local":
            print(f"{'':20} Setup: {info['setup']}")
        print()

    # Recommendations
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()

    if "gemini" in available_methods:
        print("✓ Use Gemini Flash (RECOMMENDED)")
        print("  • Fastest: <1 second per image")
        print("  • Most accurate: Full floor plan understanding")
        print("  • Cheap: $0.01 per 100 images")
        print()
        print("  Quick start:")
        print("    python3 extract_floor_plan_gemini.py --image floor5.jpg")
        print()
    elif "google" in available_methods:
        print("✓ Use Google Cloud Vision")
        print("  • Fast: 1-3 seconds per image")
        print("  • Accurate OCR")
        print("  • Moderate cost: $0.06 per 100 images")
        print()
        print("  Quick start:")
        print("    python3 extract_floor_plan_google_vision.py --image floor5.jpg")
        print()
    else:
        print("✓ Use Local OCR")
        print("  • No API setup needed")
        print("  • Works offline")
        print("  • Good accuracy")
        print("  • Slower: 30-40 seconds per image")
        print()
        print("  Quick start:")
        print("    python3 extract_floor_plan.py --image floor5.jpg --tessdata-dir ./tessdata")
        print()

    # Next steps
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print()

    missing_packages = []
    if "gemini" not in available_methods:
        if not check_package("google-generativeai")[0]:
            missing_packages.append("google-generativeai (for Gemini Flash)")
        if not os.getenv("GEMINI_API_KEY"):
            print("1. Get Gemini API key:")
            print("   https://aistudio.google.com/app/apikey")
            print()
            print("2. Install package:")
            print("   pip install google-generativeai")
            print()
            print("3. Set environment variable:")
            print("   export GEMINI_API_KEY='your-key-here'")
            print()

    print("Run comprehensive tests:")
    print("  python3 test_all_extractors.py --quick")
    print()
    print("Compare methods on single image:")
    print("  python3 test_method_compare.py")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
