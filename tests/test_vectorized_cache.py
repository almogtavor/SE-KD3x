"""
Test Vectorized Cache Unpacking Performance
============================================

This test directly compares the serial vs vectorized unpacking approach.
"""

import torch
import time
import sys
sys.path.insert(0, '/home/morg/students/almogt/sampled-kd')

from sekd.training.offline_cache import decode_ids_probs_from_block


def serial_unpack(packed_by_b, U_by_b, sen_by_b, batch_idx, pos_idx, P_total, U_max):
    """Original serial unpacking (one position at a time)."""
    ids_U = torch.zeros((P_total, U_max), dtype=torch.long)
    probs_U = torch.zeros((P_total, U_max), dtype=torch.float32)
    
    for r in range(P_total):
        b = int(batch_idx[r].item())
        p = int(pos_idx[r].item())
        U = U_by_b[b]
        sentinel = sen_by_b[b]
        packed = packed_by_b[b]
        if U == 0:
            continue
        block = packed[p * U * 3:(p + 1) * U * 3]
        ids_r, probs_r = decode_ids_probs_from_block(block, U, sentinel)
        u = ids_r.numel()
        if u > 0:
            ids_U[r, :u] = ids_r
            probs_U[r, :u] = probs_r
    
    return ids_U, probs_U


def vectorized_unpack(packed_by_b, U_by_b, sen_by_b, batch_idx, pos_idx, P_total, U_max):
    """New vectorized unpacking (batch all positions)."""
    ids_U = torch.zeros((P_total, U_max), dtype=torch.long)
    probs_U = torch.zeros((P_total, U_max), dtype=torch.float32)
    
    if U_max > 0:
        # Preallocate buffer for all packed blocks [P_total, U_max, 3]
        all_blocks = torch.zeros((P_total, U_max, 3), dtype=torch.uint8, device='cpu')
        sentinel_ids = torch.zeros(P_total, dtype=torch.int32, device='cpu')
        
        for r in range(P_total):
            b = int(batch_idx[r].item())
            p = int(pos_idx[r].item())
            U = U_by_b[b]
            sentinel_ids[r] = sen_by_b[b]
            if U == 0:
                continue
            packed = packed_by_b[b]
            block = packed[p * U * 3:(p + 1) * U * 3]
            all_blocks[r, :U, :] = block.view(U, 3)
        
        # Vectorized unpack: [P_total, U_max, 3] -> [P_total, U_max] ids and q7
        b_flat = all_blocks.to(torch.int64)  # [P_total, U_max, 3]
        x = b_flat[:, :, 0] | (b_flat[:, :, 1] << 8) | (b_flat[:, :, 2] << 16)  # [P_total, U_max]
        ids17 = x & ((1 << 17) - 1)  # 17-bit IDs
        q7 = (x >> 17) & ((1 << 7) - 1)  # 7-bit probs
        
        # Filter sentinels and zero probabilities
        sentinel_mask = ids17 != sentinel_ids.unsqueeze(1)  # [P_total, U_max]
        q7_mask = q7 > 0
        keep = sentinel_mask & q7_mask
        
        # Convert to final format
        ids_U_cpu = ids17.to(torch.int64)
        ids_U_cpu[~keep] = 0  # Zero out invalid entries
        probs_U_cpu = (q7.float() / 127.0).clamp_min(0.0)
        probs_U_cpu[~keep] = 0.0
        
        # Normalize probabilities per row
        row_sums = probs_U_cpu.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs_U_cpu = probs_U_cpu / row_sums
        
        ids_U = ids_U_cpu
        probs_U = probs_U_cpu
    
    return ids_U, probs_U


def test_performance():
    """Compare serial vs vectorized unpacking performance."""
    print("=" * 70)
    print("VECTORIZED CACHE UNPACKING PERFORMANCE TEST")
    print("=" * 70)
    
    # Setup test data
    B, L, U = 4, 512, 64  # batch, seq len, vocab samples per position
    V = 50000
    sentinel = V
    
    # Create fake packed data (uint8 format: 3 bytes per token)
    from sekd.training.offline_cache import pack_id_q7
    
    packed_by_b = []
    U_by_b = [U] * B
    sen_by_b = [sentinel] * B
    
    for b in range(B):
        # Generate random vocab samples for this batch
        ids_list = []
        cnts_list = []
        for p in range(L):
            # Random U tokens
            ids = torch.randint(0, V, (U,), dtype=torch.int32)
            # Random probabilities (will be normalized)
            q7 = torch.randint(1, 128, (U,), dtype=torch.uint8)
            ids_list.append(ids)
            cnts_list.append(q7)
        
        # Pack into format
        packed = torch.empty(L * U * 3, dtype=torch.uint8)
        for p in range(L):
            packed_block = pack_id_q7(ids_list[p], cnts_list[p])
            packed[p * U * 3:(p + 1) * U * 3] = packed_block
        
        packed_by_b.append(packed)
    
    # Create position indices
    P_total = B * L
    batch_idx = torch.repeat_interleave(torch.arange(B), L)
    pos_idx = torch.tile(torch.arange(L), (B,))
    U_max = U
    
    print(f"\nSetup:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {L}")
    print(f"  Vocab samples (U): {U}")
    print(f"  Total positions: {P_total}")
    print(f"  Packed data size: {B * L * U * 3} bytes")
    
    # Warmup
    _ = serial_unpack(packed_by_b, U_by_b, sen_by_b, batch_idx, pos_idx, P_total, U_max)
    _ = vectorized_unpack(packed_by_b, U_by_b, sen_by_b, batch_idx, pos_idx, P_total, U_max)
    
    # Benchmark serial
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        ids_serial, probs_serial = serial_unpack(packed_by_b, U_by_b, sen_by_b, batch_idx, pos_idx, P_total, U_max)
    serial_time = (time.time() - start) / num_runs
    
    # Benchmark vectorized
    start = time.time()
    for _ in range(num_runs):
        ids_vec, probs_vec = vectorized_unpack(packed_by_b, U_by_b, sen_by_b, batch_idx, pos_idx, P_total, U_max)
    vec_time = (time.time() - start) / num_runs
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"  Serial unpacking:     {serial_time*1000:.1f}ms")
    print(f"  Vectorized unpacking: {vec_time*1000:.1f}ms")
    print(f"  Speedup:              {serial_time/vec_time:.2f}x")
    
    # Verify correctness
    print(f"\n✓ CORRECTNESS CHECK:")
    print(f"  IDs match:   {torch.allclose(ids_serial.float(), ids_vec.float())}")
    print(f"  Probs match: {torch.allclose(probs_serial, probs_vec, rtol=1e-4, atol=1e-6)}")
    
    # Show some stats
    print(f"\n📈 OUTPUT STATISTICS:")
    print(f"  Non-zero IDs (serial):     {(ids_serial != 0).sum().item()}/{ids_serial.numel()}")
    print(f"  Non-zero IDs (vectorized): {(ids_vec != 0).sum().item()}/{ids_vec.numel()}")
    print(f"  Prob sums (should be ~1.0):")
    print(f"    Serial mean:     {probs_serial.sum(dim=1).mean().item():.6f}")
    print(f"    Vectorized mean: {probs_vec.sum(dim=1).mean().item():.6f}")
    
    if serial_time / vec_time > 2.0:
        print(f"\n✅ SUCCESS: Vectorized version is {serial_time/vec_time:.1f}x faster!")
    elif serial_time / vec_time > 1.0:
        print(f"\n⚠️  MARGINAL: Only {serial_time/vec_time:.1f}x speedup (expected >2x)")
    else:
        print(f"\n❌ FAILURE: Vectorized is slower! ({vec_time/serial_time:.1f}x)")
    
    print("=" * 70)


if __name__ == "__main__":
    test_performance()
