import subprocess
import os

def is_poppler_installed(poppler_path=None):
    """Check if Poppler's pdftoppm is available."""
    try:
        if poppler_path:
            pdftoppm_cmd = os.path.join(poppler_path, "pdftoppm.exe")
        else:
            pdftoppm_cmd = "pdftoppm"
        result = subprocess.run([pdftoppm_cmd, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0 or b"pdftoppm" in result.stdout or b"pdftoppm" in result.stderr
    except Exception:
        return False

# Example usage:
if __name__ == "__main__":
    poppler_bin = r"C:/poppler/poppler-24.08.0/Library/bin"
    if is_poppler_installed(poppler_bin):
        print("Poppler is installed and available.")
    else:
        print("Poppler is NOT installed or not found in the specified path.")
    # ... rest of your GUI code ...