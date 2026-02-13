# Known Limitations and Future Work

This document outlines current limitations in the GraphyloVar implementation and provides guidance for extending the codebase.

## Architecture Placeholders

### 1. Graph Convolutional Network (GCN) Layers

**Issue**: The GCN layers in training scripts are currently placeholders using Dense layers.

**Files Affected**:
- `train.py` (line 53-55)
- `train_graphylo_siamese.py` (line 130-132)
- `train_graphylovar_siamese.py` (line 32-34)

**Current Implementation**:
```python
# Placeholder GCN - currently just a Dense layer
x = Dense(gcn_units, activation="relu", name="gcn_placeholder")(x)
```

**Recommended Solution**:
Use proper GCN layers from the Spektral library:
```python
from spektral.layers import GCNConv

# Proper GCN implementation
x_gcn = GCNConv(gcn_units, activation="relu")([x, adjacency_matrix])
```

**Migration Path**:
1. Import `GCNConv` from `spektral.layers`
2. Prepare adjacency matrix in correct format (see Spektral docs)
3. Replace Dense layer with GCNConv
4. Pass both features and adjacency matrix to GCN layer
5. Test with your phylogenetic structure

### 2. Phylogenetic Adjacency Matrix

**Issue**: The adjacency matrix is currently an identity matrix, which doesn't represent actual phylogenetic relationships.

**Files Affected**:
- `config.py` (lines 174, 199)
- All training scripts that use `get_phylogenetic_adjacency_matrix()`

**Current Implementation**:
```python
def get_phylogenetic_adjacency_matrix():
    # Returns identity matrix as placeholder
    return np.eye(NUM_SPECIES)
```

**Recommended Solution**:
Implement proper phylogenetic adjacency matrix computation:

```python
import networkx as nx
import numpy as np

def get_phylogenetic_adjacency_matrix():
    """Load phylogenetic tree and compute adjacency matrix."""
    # Option 1: Load from Newick file
    from Bio import Phylo
    tree = Phylo.read("phylogenetic_tree.nwk", "newick")
    
    # Convert to networkx graph
    G = Phylo.to_networkx(tree)
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).todense()
    
    return np.array(A, dtype=np.float32)
```

**Alternative Solutions**:
1. **Precompute and save**: Compute adjacency matrix once, save as `.npy`, load at runtime
2. **Distance-based edges**: Use evolutionary distances from alignment to weight edges
3. **From topology file**: Parse phylogenetic tree topology from custom format

**Migration Path**:
1. Obtain phylogenetic tree in Newick format or similar
2. Implement tree parsing and adjacency computation
3. Cache computed adjacency matrix for performance
4. Update `get_phylogenetic_edges()` to read from tree file
5. Test that matrix dimensions match `NUM_SPECIES`

### 3. Phylogenetic Edges

**Issue**: Only a minimal set of example edges is provided.

**File**: `config.py` (line 174)

**Current Implementation**:
```python
def get_phylogenetic_edges():
    edges = [
        ("hg38", "_HP"),
        ("panTro4", "_HP"),
        # Only 2 edges as placeholder
    ]
    return edges
```

**Recommended Solution**:
Provide complete phylogenetic tree edges for all 115 species/ancestors:

```python
def get_phylogenetic_edges():
    """Load complete phylogenetic tree edges."""
    # Option 1: Hardcode full tree
    edges = [
        # Primate clade
        ("hg38", "_HP"),
        ("panTro4", "_HP"),
        ("_HP", "_HPG"),
        ("gorGor3", "_HPG"),
        # ... continue for all species
    ]
    
    # Option 2: Load from file
    import json
    with open("phylo_edges.json") as f:
        edges = json.load(f)
    
    return edges
```

**Data Format**:
```json
[
    ["hg38", "_HP"],
    ["panTro4", "_HP"],
    ["_HP", "_HPG"],
    ...
]
```

## Production Deployment Considerations

### Security Warning: Pickle Loading

**Issue**: Loading numpy arrays with `allow_pickle=True` can execute arbitrary code.

**File**: `utils.py` (line 180), `train.py` (lines 136, 141)

**Current Implementation**: Pickle loading is now explicit and disabled by default, but users must understand the risk.

**Recommendation for Production**:
1. **Only use pickle with trusted data sources**
2. **Validate data checksums before loading**
3. **Consider alternative formats**: HDF5, Zarr, or pure numpy arrays
4. **Sandbox untrusted data**: Load in isolated environment first

### Identity Matrix Warning in Production

**Issue**: Using identity adjacency matrix in production defeats the purpose of GCN.

**Recommendation**:
Add production mode check:
```python
def create_adjacency_matrices(num_samples, num_species, production=False):
    if production:
        raise ValueError(
            "Production mode requires real phylogenetic adjacency matrix. "
            "Identity matrix placeholder is not suitable for production use."
        )
    # Return identity for development/testing
    logger.warning("Using identity matrix placeholder...")
    return np.tile(np.eye(num_species), (num_samples, 1, 1))
```

## Research vs. Production Code

This codebase is structured for **research and experimentation**. The placeholders are intentional to allow:
- Testing preprocessing pipelines without full model implementation
- Experimenting with different phylogenetic representations
- Gradual implementation of complex components

### For Research Use:
✅ Use as-is for preprocessing and data exploration
✅ Use placeholders to test pipeline end-to-end
✅ Extend one component at a time

### For Production Use:
⚠️ Must implement all GCN layers properly
⚠️ Must provide real phylogenetic adjacency matrix
⚠️ Must validate all model architectures
⚠️ Must add comprehensive integration tests

## Contributing Improvements

If you implement any of these missing components, please:
1. Follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md)
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

We welcome contributions that move these placeholders toward production-ready implementations!

## References

- **Spektral GCN Documentation**: https://graphneural.network/layers/convolution/
- **NetworkX for Graphs**: https://networkx.org/
- **BioPython Phylo**: https://biopython.org/wiki/Phylo
- **MAF Format Spec**: https://genome.ucsc.edu/FAQ/FAQformat.html#format5
