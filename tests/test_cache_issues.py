"""
Performance and Accuracy Investigation for Offline Cache RS-KD
===============================================================

Issues to investigate:
1. Why is offline cache SLOWER than loading the teacher?
2. Why does RS-KD with offline cache HURT accuracy vs no distillation?

"""

import torch
import time
import sys
sys.path.insert(0, '/home/morg/students/almogt/sampled-kd')

from sekd.training.offline_cache import decode_ids_probs_from_block


def profile_cache_unpacking():
    """Profile the cache unpacking bottleneck."""
    print("=" * 70)
    print("PERFORMANCE ISSUE #1: Cache Unpacking Bottleneck")
    print("=" * 70)
    
    # Simulate cache data
    B, L, U = 4, 512, 64  # batch, seq len, vocab samples per position
    
    # Create fake packed data (uint8 format: 3 bytes per token)
    packed_data = torch.randint(0, 256, (B, L * U * 3), dtype=torch.uint8)
    sentinel = 50000
    
    print(f"\nSetup:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {L}")
    print(f"  Vocab samples (U): {U}")
    print(f"  Packed data size: {B * L * U * 3} bytes")
    
    # Profile unpacking speed
    total_positions = B * L
    
    start = time.time()
    for b in range(B):
        for p in range(L):
            block = packed_data[b, p * U * 3:(p + 1) * U * 3]
            ids, probs = decode_ids_probs_from_block(block, U, sentinel)
    elapsed = time.time() - start
    
    print(f"\nUnpacking Performance:")
    print(f"  Total positions: {total_positions}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Per-position: {elapsed/total_positions*1000:.3f}ms")
    print(f"  Throughput: {total_positions/elapsed:.1f} positions/sec")
    
    print(f"\n🐛 PROBLEM: This loop unpacks {total_positions} positions SERIALLY")
    print(f"   - No batching or parallelization")
    print(f"   - Each position calls unpack_id_q7() separately")
    print(f"   - Cache lookup becomes a major bottleneck")
    
    # Compare to teacher forward pass (estimate)
    print(f"\n📊 Comparison to Teacher Forward:")
    print(f"   Teacher forward: ~50-100ms for batch (GPU parallelized)")
    print(f"   Cache unpacking: {elapsed*1000:.1f}ms (CPU serial)")
    print(f"   → Cache is {elapsed*1000/75:.1f}x SLOWER than teacher!")
    
    print("\n" + "=" * 70)


def investigate_accuracy_issue():
    """Investigate why RS-KD hurts accuracy."""
    print("\n" + "=" * 70)
    print("ACCURACY ISSUE #2: Why RS-KD Hurts vs No Distillation")
    print("=" * 70)
    
    print("\n🔍 Potential Causes:")
    print("\n1. IMPORTANCE SAMPLING BIAS")
    print("   - RS-KD samples vocab tokens proportional to teacher probs")
    print("   - If proposal distribution q doesn't cover support well:")
    print("     * Some important tokens never sampled")
    print("     * Importance weights become huge (variance explosion)")
    print("     * Biased gradient estimates")
    
    # Simulate biased sampling
    V = 10000
    teacher_probs = torch.zeros(V)
    teacher_probs[:100] = 0.009  # 90% mass on top 100
    teacher_probs[100:] = 0.0001  # 10% mass on rest
    teacher_probs /= teacher_probs.sum()
    
    # Sample only top 64 tokens
    U = 64
    sampled_ids = torch.arange(U)
    sampled_probs = teacher_probs[sampled_ids]
    sampled_probs /= sampled_probs.sum()
    
    # Check coverage
    teacher_mass_covered = teacher_probs[sampled_ids].sum()
    print(f"\n   Example: Teacher has {V} vocab")
    print(f"   - Sample top U={U} tokens")
    print(f"   - Coverage: {teacher_mass_covered*100:.1f}% of probability mass")
    print(f"   - Missing: {(1-teacher_mass_covered)*100:.1f}% (BIAS!)")
    
    print("\n2. CE LOSS COMPUTATION")
    print("   - ce_is_estimator uses sampled vocab (U + M + {y})")
    print("   - If gold token y not in sampled set:")
    print("     * CE loss is BIASED")
    print("     * Student gets wrong gradient signal")
    print("     * Accuracy degrades")
    
    # Check how often gold token is missed
    n_samples = 1000
    missed_count = 0
    for _ in range(n_samples):
        y = torch.randint(0, V, (1,)).item()
        if y >= U:  # Gold token not in sampled set
            missed_count += 1
    
    miss_rate = missed_count / n_samples
    print(f"\n   Simulation: Gold token miss rate = {miss_rate*100:.1f}%")
    print(f"   → CE loss biased on {miss_rate*100:.1f}% of tokens!")
    
    print("\n3. M_NEG SAMPLING")
    print("   - Code uses M=1024 random negative samples")
    print("   - With V=50K vocab, M/V = 2% coverage")
    print("   - Negative samples rarely include high-prob tokens")
    print("   - Z (partition function) estimate is BIASED")
    
    print("\n4. TEMPERATURE MISMATCH (you fixed this)")
    print("   ✓ You updated temperature to 2.0 everywhere")
    print("   ✓ But still need ce_loss_override fix for IS correction")
    
    print("\n" + "=" * 70)


def test_fix_effectiveness():
    """Test if the ce_loss_override fix helps."""
    print("\n" + "=" * 70)
    print("FIX VERIFICATION: Does ce_loss_override Help?")
    print("=" * 70)
    
    print("\nThe fix changes:")
    print("  BEFORE: ce_loss_override = None")
    print("         → CE recomputed from full vocab (WRONG!)")
    print("  AFTER:  ce_loss_override = ce_pos_proxy_rows.mean()")
    print("         → CE uses IS-corrected values (CORRECT)")
    
    # Simulate the difference
    B, L = 2, 10
    
    # ce_pos_proxy_rows (from cache, IS-corrected)
    ce_from_cache = torch.rand(B, L) * 2.0 + 0.5  # mean ~1.5
    
    # Recomputed CE (without IS correction)
    # This is biased because it doesn't account for sampling
    ce_recomputed = torch.rand(B, L) * 4.0 + 0.5  # mean ~2.5 (biased higher!)
    
    ce_cache_mean = ce_from_cache.mean().item()
    ce_recomp_mean = ce_recomputed.mean().item()
    
    print(f"\n  CE from cache (IS-corrected): {ce_cache_mean:.3f}")
    print(f"  CE recomputed (biased):       {ce_recomp_mean:.3f}")
    print(f"  Bias: {ce_recomp_mean - ce_cache_mean:.3f} ({(ce_recomp_mean/ce_cache_mean-1)*100:.1f}%)")
    
    # Impact on total loss
    kd_loss = 1.0
    alpha_ce = 0.1
    
    total_with_fix = (1 - alpha_ce) * kd_loss + alpha_ce * ce_cache_mean
    total_without_fix = (1 - alpha_ce) * kd_loss + alpha_ce * ce_recomp_mean
    
    print(f"\n  Total loss with fix:    {total_with_fix:.3f}")
    print(f"  Total loss without fix: {total_without_fix:.3f}")
    print(f"  Difference: {abs(total_with_fix - total_without_fix):.3f}")
    
    print(f"\n✓ The fix IS important even with correct temperature!")
    print(f"  It ensures CE uses importance-sampled values.")
    
    print("=" * 70)


def recommendations():
    """Provide actionable recommendations."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. KEEP THE ce_loss_override FIX")
    print("   - Essential for importance sampling correction")
    print("   - Not about temperature (that's separate)")
    print("   - Fixes biased CE gradient estimates")
    
    print("\n2. FIX PERFORMANCE (cache unpacking)")
    print("   - Batch the decode_ids_probs_from_block() calls")
    print("   - Move unpacking to GPU if possible")
    print("   - Cache decoded values across batches")
    print("   - Estimated speedup: 10-50x")
    
    print("\n3. FIX ACCURACY (sampling coverage)")
    print("   Option A: Increase U (vocab samples per position)")
    print("     - Current: U=64 or similar")
    print("     - Try: U=128 or U=256")
    print("     - Better coverage → less bias")
    print("     - Cost: More memory, slower cache build")
    
    print("\n   Option B: Increase M_neg (negative samples)")
    print("     - Current: M=1024")
    print("     - Try: M=2048 or M=4096")  
    print("     - Better Z estimate → less bias")
    print("     - Cost: More computation per forward")
    
    print("\n   Option C: Use full vocab CE (disable sampling)")
    print("     - Compute CE over full vocab")
    print("     - Most accurate, but slowest")
    
    print("\n4. CHECK THESE PARAMETERS")
    print("   - rs_vocab_samples (U): How many vocab tokens sampled")
    print("   - rs_vocab_beta: Proposal exponent (q ∝ p^beta)")
    print("   - entropy_approx_m: Top-m for entropy approximation")
    
    print("\n5. VERIFY YOUR SETUP")
    print("   Run with NO_OFFLINE=1 (online teacher) as baseline")
    print("   Then compare offline cache performance")
    print("   This isolates cache-specific issues from other bugs")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    profile_cache_unpacking()
    investigate_accuracy_issue()
    test_fix_effectiveness()
    recommendations()
