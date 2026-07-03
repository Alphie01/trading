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
os.environ['TF_USE_LEGACY_KERAS'] = 'True'  # **NEW: Use legacy Keras for compatibility**

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
    """Get TensorFlow with intelligent GPU/CPU selection"""
    try:
        import tensorflow as tf
        
        print("🔧 Configuring TensorFlow with intelligent device selection...")
        
        # **STEP 1: Check GPU availability first**
        gpu_available = False
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and len(gpus) > 0:
                gpu_available = True
                print(f"🎮 {len(gpus)} GPU(s) detected: {[gpu.name for gpu in gpus]}")
                
                # **STEP 2: Try to configure GPU safely**
                try:
                    # Enable memory growth for all GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("✅ GPU memory growth enabled")
                    
                    # **Test GPU functionality**
                    with tf.device('/GPU:0'):
                        test_tensor = tf.constant([1., 2., 3.])
                        test_result = tf.reduce_sum(test_tensor).numpy()
                        print(f"✅ GPU test successful: {test_result}")
                        print("🎮 GPU mode activated!")
                        
                    return tf  # Return with GPU enabled
                    
                except Exception as gpu_error:
                    print(f"⚠️ GPU configuration failed: {gpu_error}")
                    print("🔄 Falling back to CPU mode...")
                    gpu_available = False
            else:
                print("🖥️ No GPUs detected")
                
        except Exception as gpu_check_error:
            print(f"⚠️ GPU detection error: {gpu_check_error}")
            print("🔄 Falling back to CPU mode...")
        
        # **STEP 3: Configure for CPU mode (if no GPU or GPU failed)**
        if not gpu_available:
            print("🖥️ Configuring TensorFlow for CPU-only mode...")
            
            try:
                # Disable GPU access
                tf.config.set_visible_devices([], 'GPU')
                print("✅ GPU access disabled - CPU-only mode")
            except Exception as gpu_disable_error:
                print(f"⚠️ GPU disable warning: {gpu_disable_error}")
            
            # **Conservative threading for CPU**
            try:
                tf.config.threading.set_inter_op_parallelism_threads(2)
                tf.config.threading.set_intra_op_parallelism_threads(4)
                print("✅ CPU threading optimized")
            except Exception as threading_error:
                print(f"⚠️ Threading config warning: {threading_error}")
        
        # **STEP 4: Set general TensorFlow optimizations**
        try:
            # Enable mixed precision for better performance (if supported)
            if gpu_available:
                try:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    print("✅ Mixed precision enabled (GPU)")
                except:
                    print("⚠️ Mixed precision not supported")
            
            # Set device policy
            tf.config.experimental.set_device_policy('silent_for_int32')
            print("✅ Device policy configured")
            
        except Exception as config_error:
            print(f"⚠️ General config warning: {config_error}")
        
        # **STEP 5: Final functionality test**
        try:
            test_tensor = tf.constant([1., 2., 3.])
            test_computation = tf.reduce_sum(test_tensor)
            final_result = test_computation.numpy()
            
            # Determine which device was actually used
            device_name = test_tensor.device
            if 'GPU' in device_name:
                print(f"🎮 TensorFlow configured successfully with GPU: {final_result}")
            else:
                print(f"🖥️ TensorFlow configured successfully with CPU: {final_result}")
                
        except Exception as test_error:
            print(f"⚠️ TensorFlow test warning: {test_error}")
        
        print("✅ TensorFlow intelligent configuration complete")
        return tf
        
    except ImportError as e:
        print(f"❌ TensorFlow import error: {e}")
        return None
    except Exception as e:
        print(f"❌ TensorFlow configuration error: {e}")
        # **Fallback: Return basic TensorFlow**
        try:
            import tensorflow as tf
            print("🔄 Using basic TensorFlow configuration")
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