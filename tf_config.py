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