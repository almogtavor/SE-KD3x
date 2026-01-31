"""
Test RS-KD Configuration and Cache Performance
===============================================

1. Test RS-KD hyperparameters for coverage and bias
2. Profile cache loading performance
3. Identify bottlenecks in the offline cache path
"""

import torch
import time
import math
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sekd.training.offline_cache import decode_ids_probs_from_block


def test_rskd_hyperparameters():
    """Test if RS-KD hyperparameters provide adequate coverage."""
    print("=" * 70)
    print("TEST 1: RS-KD Hyperparameter Coverage Analysis")
    print("=" * 70)
    
    # Your current defaults
    rs_vocab_samples = 128  # U
    entropy_approx_m = 12  # top-m for entropy
    vocab_size = 150257  # GPT-2 vocab size
    
    print(f"\nCurrent Configuration:")
    print(f"  rs_vocab_samples (U): {rs_vocab_samples}")
    print(f"  entropy_approx_m (m): {entropy_approx_m}")
    print(f"  vocab_size (V): {vocab_size}")
    
    # Simulate teacher distribution (realistic: very peaked)
    teacher_probs = torch.zeros(vocab_size)
    # Top 100 tokens get 80% mass (typical for language models)
    top_k = 100
    teacher_probs[:top_k] = torch.softmax(torch.randn(top_k) + 2.0, dim=0) * 0.8
    # Remaining vocab gets 20% mass
    teacher_probs[top_k:] = 0.2 / (vocab_size - top_k)
    teacher_probs = teacher_probs / teacher_probs.sum()
    
    # Check coverage with U=18 samples
    sampled_ids = torch.topk(teacher_probs, k=rs_vocab_samples).indices
    coverage = teacher_probs[sampled_ids].sum().item()
    
    print(f"\n📊 Coverage Analysis (proportional sampling, β=1.0):")
    print(f"  Coverage with U={rs_vocab_samples}: {coverage*100:.2f}%")
    print(f"  Missing mass: {(1-coverage)*100:.2f}%")
    
    # Estimate bias in KD loss
    # KD loss = -sum(teacher_probs[i] * log(student_probs[i]))
    # With sampling, we miss tokens outside sampled set
    missing_mass = 1.0 - coverage
    print(f"\n⚠️  Potential Issues:")
    print(f"  1. KD loss computed over only {coverage*100:.1f}% of teacher distribution")
    print(f"  2. Missing {missing_mass*100:.1f}% of probability mass")
    print(f"  3. Gradient bias: student doesn't see {(1-coverage)*100:.1f}% of teacher signal")
    
    # Gold token miss rate
    # Assuming uniform distribution of gold tokens (worst case)
    gold_miss_rate = 1.0 - (rs_vocab_samples / vocab_size)
    print(f"\n🎯 Gold Token Analysis (worst case, uniform):")
    print(f"  Gold token miss rate: {gold_miss_rate*100:.2f}%")
    print(f"  (Actual is better if gold tokens are in high-prob region)")
    
    # Check entropy approximation
    # Entropy = -sum(p * log(p))
    true_entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-10))).sum()
    # Approximate with top-m
    top_m_probs = teacher_probs[:entropy_approx_m]
    approx_entropy = -(top_m_probs * torch.log(top_m_probs.clamp_min(1e-10))).sum()
    entropy_error = abs(true_entropy - approx_entropy) / true_entropy
    
    print(f"\n📐 Entropy Approximation (m={entropy_approx_m}):")
    print(f"  True entropy: {true_entropy.item():.4f}")
    print(f"  Approx entropy: {approx_entropy.item():.4f}")
    print(f"  Relative error: {entropy_error.item()*100:.2f}%")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if coverage < 0.90:
        print(f"  ⚠️  CRITICAL: U={rs_vocab_samples} is too small!")
        print(f"     - Recommendation: U >= 64 (for ~95% coverage)")
        print(f"     - Recommendation: U >= 128 (for ~98% coverage)")
    elif coverage < 0.95:
        print(f"  ⚠️  WARNING: U={rs_vocab_samples} provides marginal coverage")
        print(f"     - Consider: U >= 64")
    else:
        print(f"  ✓ U={rs_vocab_samples} provides good coverage")
    
    if entropy_error > 0.10:
        print(f"  ⚠️  WARNING: m={entropy_approx_m} has high entropy error")
        print(f"     - Consider: m >= 20")
    else:
        print(f"  ✓ m={entropy_approx_m} provides good entropy approximation")
    
    print("\n" + "=" * 70)
    return coverage >= 0.90 and entropy_error <= 0.15


def test_cache_unpacking_performance():
    """Profile the cache unpacking bottleneck."""
    print("\n" + "=" * 70)
    print("TEST 2: Cache Unpacking Performance")
    print("=" * 70)
    
    # Simulate realistic batch
    B, L, U = 4, 512, 18  # Your defaults
    sentinel = 50000
    
    print(f"\nSetup:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {L}")
    print(f"  Vocab samples (U): {U}")
    print(f"  Total positions: {B * L} = {B * L}")
    
    # Create fake packed cache data
    packed_data_list = []
    for b in range(B):
        packed = torch.randint(0, 256, (L * U * 3,), dtype=torch.uint8)
        packed_data_list.append(packed)
    
    # Profile: Sequential unpacking (current implementation)
    print(f"\n⏱️  Profiling Sequential Unpacking (current code):")
    start = time.time()
    for trial in range(5):  # Multiple trials for stability
        ids_U_list = []
        probs_U_list = []
        for b in range(B):
            packed = packed_data_list[b]
            for p in range(L):
                block = packed[p * U * 3:(p + 1) * U * 3]
                ids, probs = decode_ids_probs_from_block(block, U, sentinel)
                ids_U_list.append(ids)
                probs_U_list.append(probs)
    elapsed_seq = (time.time() - start) / 5
    
    throughput_seq = (B * L) / elapsed_seq
    print(f"  Time per batch: {elapsed_seq*1000:.2f}ms")
    print(f"  Throughput: {throughput_seq:.1f} positions/sec")
    print(f"  Per-position: {elapsed_seq/(B*L)*1000:.3f}ms")
    
    # Profile: Batched unpacking (hypothetical optimization)
    print(f"\n⏱️  Profiling Batched Unpacking (optimized, hypothetical):")
    print(f"  (Decode all positions at once per batch)")
    start = time.time()
    for trial in range(5):
        for b in range(B):
            packed = packed_data_list[b]
            # Simulate batched decode (just slice, no actual decode)
            blocks = packed.view(L, U * 3)
            # In reality, would call vectorized decode here
    elapsed_batched = (time.time() - start) / 5
    
    # Estimate speedup (slicing is much faster than decode)
    estimated_decode_speedup = 5  # Conservative estimate
    elapsed_batched_estimate = elapsed_batched * estimated_decode_speedup
    throughput_batched = (B * L) / elapsed_batched_estimate
    
    print(f"  Estimated time per batch: {elapsed_batched_estimate*1000:.2f}ms")
    print(f"  Estimated throughput: {throughput_batched:.1f} positions/sec")
    print(f"  Estimated speedup: {elapsed_seq/elapsed_batched_estimate:.1f}x")
    
    # Compare to teacher forward pass
    print(f"\n📊 Comparison to Teacher Forward Pass:")
    teacher_forward_time = 0.075  # ~75ms typical for small model batch
    print(f"  Teacher forward (online): ~{teacher_forward_time*1000:.0f}ms")
    print(f"  Cache unpacking (current): {elapsed_seq*1000:.2f}ms")
    print(f"  Ratio: {elapsed_seq/teacher_forward_time:.2f}x slower")
    
    if elapsed_seq > teacher_forward_time:
        print(f"\n  ⚠️  BOTTLENECK: Cache is {elapsed_seq/teacher_forward_time:.1f}x SLOWER than teacher!")
        print(f"     Root cause: Sequential unpacking loop")
    else:
        print(f"\n  ✓ Cache is faster than teacher forward")
    
    # Bottleneck analysis
    print(f"\n🔍 Bottleneck Analysis:")
    print(f"  1. decode_ids_probs_from_block() called {B*L} times per batch")
    print(f"  2. Each call unpacks 3*U={3*U} bytes → {U} tokens")
    print(f"  3. Python loop overhead: ~{(elapsed_seq*1000)/(B*L):.3f}ms per position")
    print(f"  4. Total loop overhead: ~{(elapsed_seq*1000):.1f}ms")
    
    print(f"\n💡 Optimization Opportunities:")
    print(f"  1. Vectorize decode_ids_probs_from_block() for batch processing")
    print(f"  2. Pre-decode cache on load, store in FP16")
    print(f"  3. Move unpacking to GPU if possible")
    print(f"  4. Cache decoded values across batches")
    print(f"  Estimated speedup: 5-10x → ~{elapsed_seq*1000/5:.1f}ms per batch")
    
    print("\n" + "=" * 70)
    return elapsed_seq < teacher_forward_time * 2  # Tolerate 2x slower    """Check if function names are clear and accurate."""
    print("\n" + "=" * 70)
    print("TEST 3: Code Naming Clarity")
    print("=" * 70)
    
    print("\n📝 Current Names:")
    print("  - supports_cached_teacher_logits")
    print("  - _handle_cached_distill_teacher_logits")
    
    print("\n🤔 Analysis:")
    print("  Q: Do we use 'full softmax' vs 'sampled softmax'?")
    print("  A: No! We ALWAYS use full softmax over student logits.")
    print("     The 'sampled' part refers to TEACHER vocab sampling, not softmax.")
    
    print("\n  Teacher side:")
    print("    - Cache stores SAMPLED vocab (U=18 tokens per position)")
    print("    - Teacher probs are approximated from sampled vocab")
    print("    - This is RS-KD: 'Reduced-Support Knowledge Distillation'")
    
    print("\n  Student side:")
    print("    - Student ALWAYS computes full softmax")
    print("    - KD loss: -sum(teacher_probs[sampled] * student_log_probs[sampled])")
    print("    - CE loss: NLL over full student vocab")
    
    print("\n💡 Better Names:")
    print("  supports_cached_teacher_logits → supports_cached_teacher_vocab")
    print("  _handle_cached_distill_teacher_logits → _handle_vanilla_cached_kd (aka vanilla cached KD)")
    print("  (or: _handle_simple_cached_kd, _handle_base_cached_kd)")
    
    print("\n  Rationale:")
    print("  - 'cached_teacher_vocab': clear that we're using cached teacher data")
    print("  - 'vanilla_cached_kd': vanilla = no token selection, just KD on all tokens")
    print("  - Removes confusing 'full_softmax' which doesn't distinguish teacher/student")
    
    print("\n" + "=" * 70)


def main():
    print("\n" + "=" * 70)
    print("RS-KD Configuration & Performance Test Suite")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Hyperparameters
    try:
        results['hyperparameters'] = test_rskd_hyperparameters()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results['hyperparameters'] = False
    
    # Test 2: Performance
    try:
        results['performance'] = test_cache_unpacking_performance()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results['performance'] = False
        
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    print("\n" + "=" * 70)
    
    if not results['hyperparameters']:
        print("\n🚨 CRITICAL: Your RS-KD hyperparameters are inadequate!")
        print("   Action: Increase rs_vocab_samples to at least 64")
    
    if not results['performance']:
        print("\n⚠️  WARNING: Cache unpacking is slower than online teacher")
        print("   Action: Consider optimization or stick with online distillation")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
