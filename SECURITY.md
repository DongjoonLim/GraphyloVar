# Security Policy

## Critical Security Update: TensorFlow Vulnerabilities

### ⚠️ URGENT: TensorFlow 2.5.0 Has Critical Vulnerabilities

**Original Version:** TensorFlow 2.5.0  
**Updated Version:** TensorFlow 2.12.1+  
**Date Identified:** 2024  
**Severity:** CRITICAL

The original TensorFlow 2.5.0 dependency has **150+ known security vulnerabilities** including:
- Buffer overflows and heap corruption
- Null pointer dereferences
- Integer overflows
- Code injection vulnerabilities
- Out-of-bounds read/write operations
- Division by zero errors
- Segmentation faults in multiple operations

### Affected CVEs (Sample)

Major vulnerability categories fixed in TensorFlow 2.12.1:

1. **Buffer Overflow & Memory Corruption**
   - CVE-2022-XXXXX: Heap-buffer-overflow in AvgPoolGrad
   - CVE-2022-XXXXX: Buffer overflow in CONV_3D_TRANSPOSE on TFLite
   - CVE-2022-XXXXX: OOB write in scatter_nd in TF Lite
   - CVE-2022-XXXXX: Double free in Fractional(Max/Avg)Pool

2. **Null Pointer Vulnerabilities**
   - CVE-2022-XXXXX: Null Pointer Error in TensorArrayConcatV2
   - CVE-2022-XXXXX: Null Pointer Error in SparseSparseMaximum
   - CVE-2022-XXXXX: Null Pointer Error in QuantizedMatMulWithBiasAndDequantize
   - CVE-2022-XXXXX: Null Pointer Error in LookupTableImportV2
   - CVE-2022-XXXXX: Null Pointer Error in RandomShuffle with XLA

3. **Integer & Arithmetic Issues**
   - CVE-2022-XXXXX: Integer overflow in EditDistance
   - CVE-2022-XXXXX: Floating Point Exception in AudioSpectrogram
   - CVE-2022-XXXXX: Floating Point Exception in AvgPoolGrad with XLA
   - CVE-2022-XXXXX: Division by zero in multiple operations

4. **Code Injection**
   - CVE-2022-XXXXX: Code injection in saved_model_cli

5. **Segmentation Faults**
   - CVE-2022-XXXXX: Segfault in array_ops.upper_bound
   - CVE-2022-XXXXX: Segfault in Bincount with XLA
   - CVE-2022-XXXXX: Segfault in tf.raw_ops.Print
   - CVE-2022-XXXXX: Segmentation fault in tfg-translate

6. **Out-of-Bounds Access**
   - CVE-2022-XXXXX: Out-of-Bounds Read in DynamicStitch
   - CVE-2022-XXXXX: Out-of-Bounds Read in GRUBlockCellGrad
   - CVE-2022-XXXXX: OOB read in Gather_nd in TF Lite
   - CVE-2022-XXXXX: Out of bounds write in grappler

### Impact Assessment

**Severity:** CRITICAL  
**CVSS Score Range:** 5.0 - 9.0  
**Exploitability:** Many vulnerabilities can be triggered by malicious input data

**Risk to GraphyloVar Users:**
- ✅ **Low risk if processing only trusted data** (your own genomic datasets)
- ⚠️ **High risk if processing untrusted user-uploaded models or data**
- ⚠️ **High risk in production environments with external inputs**

### Required Action

**All users must upgrade to TensorFlow 2.12.1 or later.**

## Migration Guide: TensorFlow 2.5.0 → 2.12.1+

### Breaking Changes

TensorFlow 2.12.1 introduces several API changes from 2.5.0:

#### 1. Keras API Changes
```python
# TF 2.5.0 (OLD)
from tensorflow.keras.optimizers import Adam

# TF 2.12+ (NEW) 
from tensorflow.keras.optimizers import Adam  # Still works, but legacy
from tensorflow.keras.optimizers.legacy import Adam  # Explicit legacy
from tensorflow.keras.optimizers.experimental import Adam  # New optimizer
```

#### 2. Model Saving/Loading
```python
# TF 2.5.0 (OLD)
model.save('model.h5')  # H5 format

# TF 2.12+ (RECOMMENDED)
model.save('model')  # SavedModel format (directory)
model.save('model.keras')  # New Keras v3 format
```

#### 3. Deprecated Functions Removed
- `tf.compat.v1` functions may behave differently
- Some experimental APIs have been stabilized or removed

### Migration Steps

#### Step 1: Update Dependencies

```bash
# Update requirements
pip install tensorflow>=2.12.1

# Or with conda
conda install tensorflow>=2.12.1
```

#### Step 2: Test Your Code

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
assert tf.__version__ >= "2.12.1", "TensorFlow version too old!"
```

#### Step 3: Run Model Tests

```bash
# Test preprocessing
python preprocess_graphs.py --help

# Test training with small data
python train.py --epochs 1 --batch_size 8 [other args]
```

#### Step 4: Address Deprecation Warnings

Run your code and fix any deprecation warnings:
```bash
python -W default::DeprecationWarning train.py [args]
```

### Known Compatibility Issues

#### Issue 1: Optimizer State Loading
**Problem:** Models saved with TF 2.5.0 optimizers may not load properly  
**Solution:** Retrain model or use legacy optimizer:
```python
from tensorflow.keras.optimizers.legacy import Adam
optimizer = Adam(learning_rate=0.001)
```

#### Issue 2: Mixed Precision Training
**Problem:** Mixed precision API changed  
**Solution:** Update to new API:
```python
# TF 2.5.0
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

# TF 2.12+
policy = tf.keras.mixed_precision.Policy('mixed_float16')
```

#### Issue 3: Custom Layers
**Problem:** Custom layer serialization changed  
**Solution:** Implement `get_config()` properly:
```python
class CustomLayer(tf.keras.layers.Layer):
    def get_config(self):
        config = super().get_config()
        config.update({'custom_param': self.custom_param})
        return config
```

### Testing Checklist

After migration, verify:
- [ ] All dependencies install without conflicts
- [ ] Preprocessing scripts run without errors
- [ ] Training completes successfully
- [ ] Model can be saved and loaded
- [ ] Predictions match expected format
- [ ] No deprecation warnings (or all addressed)
- [ ] Tests pass (if you have tests)

### Rollback Plan

If migration fails, you can temporarily continue with TensorFlow 2.5.0:

⚠️ **WARNING:** Only use TensorFlow 2.5.0 in isolated, trusted environments:
- Do NOT expose to untrusted data
- Do NOT use in production
- Do NOT process user-uploaded files
- Use ONLY for offline research with your own data

```bash
# Rollback (NOT RECOMMENDED)
pip install tensorflow==2.5.0

# Run in isolated environment
docker run --rm -it tensorflow/tensorflow:2.5.0 /bin/bash
```

## Reporting Security Issues

If you discover a security vulnerability in GraphyloVar code (not TensorFlow), please report it to:
- **GitHub Security Advisories**: Use the "Security" tab
- **Email**: [Repository maintainer's email if available]

**Please DO NOT open public issues for security vulnerabilities.**

## Security Best Practices

### For Developers

1. **Keep Dependencies Updated**
   ```bash
   pip list --outdated
   pip install --upgrade tensorflow numpy pandas
   ```

2. **Use Virtual Environments**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Scan for Vulnerabilities**
   ```bash
   pip install safety
   safety check
   ```

4. **Pin Dependencies in Production**
   ```
   tensorflow==2.12.1  # Exact version
   numpy==1.23.5       # Not >= which could pull vulnerable versions
   ```

### For Users

1. **Process Only Trusted Data**
   - Use your own genomic datasets
   - Verify data sources
   - Validate input formats

2. **Isolate Environments**
   - Use containers (Docker/Singularity)
   - Use separate conda environments
   - Don't mix with other projects

3. **Regular Updates**
   - Check for security advisories monthly
   - Update dependencies quarterly
   - Monitor GitHub security alerts

4. **Data Validation**
   - Validate file formats before processing
   - Check file sizes (avoid malicious large files)
   - Sanitize paths (avoid directory traversal)

## Security Checklist for Production

Before deploying GraphyloVar in production:

- [ ] TensorFlow >= 2.12.1 installed
- [ ] All dependencies scanned for vulnerabilities
- [ ] Input validation implemented
- [ ] File upload restrictions in place
- [ ] Resource limits configured (memory, CPU)
- [ ] Error messages sanitized (no stack traces to users)
- [ ] Logging enabled for security events
- [ ] Regular backups configured
- [ ] Monitoring and alerting set up
- [ ] Incident response plan documented

## Version Support

| TensorFlow Version | Support Status | Security Status | Recommendation |
|-------------------|----------------|-----------------|----------------|
| 2.5.0 | ❌ End of Life | ⚠️ VULNERABLE | Upgrade immediately |
| 2.6.x - 2.11.x | ❌ End of Life | ⚠️ VULNERABLE | Upgrade to 2.12.1+ |
| 2.12.1+ | ✅ Supported | ✅ Patched | Recommended |
| 2.13.x+ | ✅ Supported | ✅ Patched | Also acceptable |

## Additional Resources

- [TensorFlow Security Advisories](https://github.com/tensorflow/tensorflow/security/advisories)
- [CVE Database](https://cve.mitre.org/)
- [National Vulnerability Database](https://nvd.nist.gov/)
- [TensorFlow Release Notes](https://github.com/tensorflow/tensorflow/releases)

## Change Log

- **2024-12-XX**: Updated from TensorFlow 2.5.0 to 2.12.1+ (150+ CVEs patched)
- **2024-12-XX**: Created SECURITY.md with migration guide
- **2024-12-XX**: Added security best practices documentation

---

**Last Updated:** December 2024  
**Next Review:** March 2025 (or upon new CVE discovery)
