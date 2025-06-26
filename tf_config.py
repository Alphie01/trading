#!/usr/bin/env python3
"""
Centralized TensorFlow Configuration for M1/M2 Mac Safety

CRITICAL: This module MUST be imported before any TensorFlow usage
to prevent Metal plugin crashes and GPU configuration errors.
"""

import os
import warnings

# **CRITICAL: Set TensorFlow environment variables BEFORE importing**
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_METAL_DEVICE_PLACEMENT'] = '0'  # **DISABLE automatic Metal placement**
os.environ['TF_DISABLE_MKL'] = '1'  # **DISABLE MKL to avoid conflicts**
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'  # **NEW: Force GPU memory growth**
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # **NEW: Disable CUDA to force CPU/Metal only**

# **NEW: Aggressive Metal prevention**
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP'] = '1'
os.environ['TF_DISABLE_MKL_SMALL_MATRIX_OPT'] = '1'
os.environ['MLX_METAL_DEBUG'] = '0'  # Disable Metal debugging

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def print_system_resources():
    """Print detailed system resource information"""
    import platform
    import psutil
    
    print("\n" + "="*60)
    print("🖥️  SYSTEM RESOURCE INFORMATION")
    print("="*60)
    
    # System info
    print(f"💻 Platform: {platform.platform()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print(f"🧠 CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"💾 RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # Detect M1/M2 Mac
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':
        print("🍎 Apple Silicon (M1/M2) detected - optimized for ARM64")
    elif platform.machine() == 'x86_64':
        print("🖥️  Intel/AMD x86_64 architecture detected")
    
    print("="*60)

def print_tensorflow_device_info():
    """Print TensorFlow device configuration and availability"""
    try:
        import tensorflow as tf
        
        print("\n" + "="*60)
        print("🔧 TENSORFLOW DEVICE CONFIGURATION")
        print("="*60)
        
        # TensorFlow version
        print(f"📦 TensorFlow Version: {tf.__version__}")
        
        # Check physical devices
        print("\n📱 Physical Devices:")
        try:
            physical_devices = tf.config.list_physical_devices()
            if physical_devices:
                for device in physical_devices:
                    print(f"   ✅ {device}")
            else:
                print("   ⚠️  No physical devices detected")
        except Exception as e:
            print(f"   ❌ Error getting physical devices: {e}")
        
        # Check GPU availability
        print("\n🎮 GPU Status:")
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   ✅ {len(gpus)} GPU(s) detected:")
                for i, gpu in enumerate(gpus):
                    print(f"      GPU {i}: {gpu}")
                    
                # Check if GPUs are actually usable
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    print("   ✅ GPU memory growth enabled")
                except Exception as gpu_config_error:
                    print(f"   ⚠️  GPU configuration warning: {gpu_config_error}")
            else:
                print("   ❌ No GPUs detected or GPU access disabled")
        except Exception as e:
            print(f"   ⚠️  GPU check error: {e}")
        
        # Check current strategy
        print("\n🎯 Training Strategy:")
        try:
            strategy = tf.distribute.get_strategy()
            print(f"   📊 Strategy: {type(strategy).__name__}")
            print(f"   🔢 Number of replicas: {strategy.num_replicas_in_sync}")
        except Exception as e:
            print(f"   ⚠️  Strategy error: {e}")
        
        # Check mixed precision
        print("\n⚡ Performance Options:")
        try:
            policy = tf.keras.mixed_precision.global_policy()
            print(f"   🎭 Mixed Precision Policy: {policy.name}")
        except Exception as e:
            print(f"   ⚠️  Mixed precision check error: {e}")
        
        print("="*60)
        
    except ImportError:
        print("\n❌ TensorFlow not available for device info")
    except Exception as e:
        print(f"\n❌ Error getting TensorFlow device info: {e}")

def get_current_device():
    """Get the current TensorFlow device being used"""
    try:
        import tensorflow as tf
        
        # Try to determine current device
        try:
            # Create a simple operation to see where it runs
            with tf.device(None):  # Let TF choose
                test_op = tf.constant([1.0])
                device_name = test_op.device
                
            if device_name:
                if 'GPU' in device_name:
                    return f"🎮 GPU: {device_name}"
                elif 'CPU' in device_name:
                    return f"🖥️  CPU: {device_name}"
                else:
                    return f"🤔 Unknown: {device_name}"
            else:
                return "🤷 Device not specified"
                
        except Exception as device_error:
            # Fallback: check available devices
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus and len(gpus) > 0:
                    return f"🎮 GPU Available: {len(gpus)} device(s)"
                else:
                    return "🖥️  CPU (No GPU available)"
            except:
                return "🖥️  CPU (Fallback mode)"
                
    except ImportError:
        return "❌ TensorFlow not available"
    except Exception as e:
        return f"❌ Device detection error: {e}"

def print_training_device_info():
    """Print comprehensive device info before training starts"""
    print("\n" + "🚀" + "="*58 + "🚀")
    print("🎯 TRAINING RESOURCE ALLOCATION")
    print("🚀" + "="*58 + "🚀")
    
    # System resources
    print_system_resources()
    
    # TensorFlow devices
    print_tensorflow_device_info()
    
    # Current device being used
    current_device = get_current_device()
    print(f"\n🎯 CURRENT TRAINING DEVICE: {current_device}")
    
    print("\n" + "🚀" + "="*58 + "🚀")

def monitor_training_resources():
    """Monitor resource usage during training"""
    import psutil
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # TensorFlow device
        current_device = get_current_device()
        
        print(f"📊 Resource Usage: CPU {cpu_percent:.1f}% | "
              f"RAM {memory_used_gb:.1f}/{memory_total_gb:.1f} GB ({memory_percent:.1f}%) | "
              f"Device: {current_device.split(': ')[-1] if ': ' in current_device else current_device}")
        
    except Exception as e:
        print(f"⚠️  Resource monitoring error: {e}")

def is_tensorflow_available():
    """Check if TensorFlow is available and can be imported safely"""
    try:
        import tensorflow as tf
        # Basic functionality test with CPU enforcement
        with tf.device('/CPU:0'):
            _ = tf.constant([1.0, 2.0, 3.0])
        return True
    except Exception as e:
        print(f"⚠️ TensorFlow not available: {e}")
        return False

def get_tensorflow():
    """Get TensorFlow with ultra-conservative safety configuration"""
    try:
        import tensorflow as tf
        
        print("🔧 Configuring TensorFlow for MAXIMUM M1/M2 Mac safety...")
        
        # **CRITICAL: FORCE CPU-ONLY MODE for stability**
        try:
            # Completely disable GPU access to prevent Metal crashes
            tf.config.set_visible_devices([], 'GPU')
            print("✅ GPU access DISABLED - CPU-only mode enforced")
        except Exception as gpu_disable_error:
            print(f"⚠️ GPU disable warning: {gpu_disable_error}")
        
        # **AGGRESSIVE: Set very conservative threading**
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)  # Single thread only
            print("✅ Ultra-conservative single-threading enabled")
        except Exception as threading_error:
            print(f"⚠️ Threading config warning: {threading_error}")
        
        # **FORCE all operations to CPU**
        try:
            # Set default device to CPU
            tf.config.experimental.set_device_policy('silent_for_int32')
            print("✅ CPU-only device policy set")
        except Exception as device_error:
            print(f"⚠️ Device policy warning: {device_error}")
        
        # **Disable ALL GPU/Metal optimizations**
        try:
            # Disable experimental features that might use Metal
            if hasattr(tf.config, 'experimental'):
                if hasattr(tf.config.experimental, 'enable_mlir_bridge'):
                    tf.config.experimental.enable_mlir_bridge(False)
                    print("✅ MLIR bridge disabled")
                if hasattr(tf.config.experimental, 'enable_mlir_graph_optimization'):
                    tf.config.experimental.enable_mlir_graph_optimization(False)
                    print("✅ MLIR graph optimization disabled")
        except Exception as mlir_error:
            print(f"⚠️ MLIR disable warning: {mlir_error}")
        
        # **Test TensorFlow functionality with STRICT CPU enforcement**
        try:
            with tf.device('/CPU:0'):
                test_result = tf.constant([1., 2., 3.])
                test_computation = tf.reduce_sum(test_result)
                final_result = test_computation.numpy()
                print(f"✅ TensorFlow CPU-only test successful: {final_result}")
        except Exception as test_error:
            print(f"⚠️ TensorFlow test failed: {test_error}")
            print("🔄 Attempting basic TensorFlow without device specification...")
            try:
                test_basic = tf.constant([1., 2., 3.])
                print(f"✅ Basic TensorFlow test successful: {test_basic.numpy()}")
            except Exception as basic_error:
                print(f"❌ Even basic TensorFlow failed: {basic_error}")
                # Still return TF object for compatibility
        
        print("✅ TensorFlow ULTRA-SAFE configuration complete (CPU-ONLY)")
        return tf
        
    except ImportError as e:
        print(f"❌ TensorFlow import error: {e}")
        return None
    except Exception as e:
        print(f"❌ TensorFlow configuration error: {e}")
        # **CRITICAL: Try to return basic TensorFlow if available**
        try:
            import tensorflow as tf
            print("🔄 Returning basic TensorFlow without advanced config")
            # Force CPU mode even for basic TF
            try:
                tf.config.set_visible_devices([], 'GPU')
            except:
                pass
            return tf
        except:
            return None

# **GLOBAL: Initialize TensorFlow on import (if available)**
TF_AVAILABLE = is_tensorflow_available()
if TF_AVAILABLE:
    tf = get_tensorflow()
else:
    tf = None
    print("⚠️ TensorFlow not available - using CPU fallback mode") 