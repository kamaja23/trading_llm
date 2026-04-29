#!/usr/bin/env python3
"""
Verification script to check if the environment is set up correctly.
Run this before starting the training pipeline.
"""

import sys
import importlib

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} installed")
        return True
    except ImportError:
        print(f"❌ {package_name} NOT installed")
        return False

def check_gpu():
    """Check if GPU is available and verify RTX 2070 compatibility"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            
            # Check for RTX 2070 specific optimizations
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"   VRAM: {gpu_props.total_mem / 1024**3:.1f} GB")
            print(f"   Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            # RTX 2070 has compute capability 7.5
            if gpu_props.major == 7 and gpu_props.minor == 5:
                print("   ✅ RTX 2070 detected - optimized settings enabled (FP16)")
            elif gpu_props.major >= 7:
                print(f"   ✅ GPU supports mixed precision (FP16)")
            else:
                print("   ⚠️  GPU may not support FP16 efficiently")
            
            # Check CUDA version
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Recommend CUDA 11.8 for RTX 2070
            if torch.version.cuda and "11.8" in torch.version.cuda:
                print("   ✅ CUDA 11.8 detected (recommended for RTX 2070)")
            else:
                print("   ⚠️  For RTX 2070, CUDA 11.8 is recommended")
                print("      Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            
            return True
        else:
            print("⚠️  No GPU detected (CPU will be used, training will be slower)")
            return True  # Not a failure, just slower
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Trading LLM Hello World - Environment Verification")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check Python version
    print("Checking Python version...")
    if not check_python_version():
        all_good = False
    print()
    
    # Check required packages
    print("Checking required packages...")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yfinance", "yfinance"),
        ("ta", "ta"),
        ("tqdm", "tqdm"),
    ]
    
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_good = False
    print()
    
    # Check GPU
    print("Checking GPU availability...")
    check_gpu()
    print()
    
    # Final verdict
    print("=" * 60)
    if all_good:
        print("✅ All checks passed! You're ready to go.")
        print()
        print("Next steps:")
        print("  1. Run: python src/01_generate_training_data.py")
        print("  2. Run: python src/02_train_model.py")
        print("  3. Run: python src/03_test_model.py")
        print("  4. Run: python src/04_interactive_inference.py")
        print()
        print("Or use the quick start script: ./quick_start.sh")
    else:
        print("❌ Some checks failed. Please install missing packages:")
        print()
        print("  pip install -r requirements.txt")
        print()
    print("=" * 60)
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
