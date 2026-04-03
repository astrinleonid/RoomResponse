"""
analyze_overlap.py
Deep analysis of why only 12 modes overlap between STANDARD_BANDS and EXTENDED_BANDS
on Belarus Scenario 40.

Usage:
    cd D:/repos/PianoidInstall/PianoidCore && .venv/Scripts/python D:/repos/RoomResponse/ESPRIT/analyze_overlap.py
"""
import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from esprit_core import (
    esprit_modal_identification, build_hankel_matrix, build_multichannel_hankel,
    esprit_poles, validate_conjugate_pairs, poles_to_modal_params,
    filter_poles, filter_poles_by_radius
)
from band_processing import (
    STANDARD_BANDS, EXTENDED_BANDS, process_band, FrequencyBand
)


def load_data():
    """Load Belarus Scenario 40 data."""
    data_dir = Path("D:/repos/RoomResponse/piano/Belarus-Scenario40-Measuement1/averaged_responses")
    channels = []
    for i in range(6):
        ch = np.load(data_dir / f"average_ch{i}.npy")
        channels.append(ch)
    signals = np.column_stack(channels)
    return signals


def run_esprit_multiband(signals, fs, bands, global_model_order, use_gpu=True,
                         window_length=2000):
    """Run ESPRIT per band, return detailed per-mode info."""
    results = []

    for band in bands:
        band_data, fs_band, metadata = process_band(signals, fs, band, apply_preemphasis=True)
        mo = band.model_order if band.model_order is not None else global_model_order
        wl = min(window_length, len(band_data) // 2)

        try:
            result = esprit_modal_identification(
                band_data, fs=fs_band, model_order=mo, window_length=wl,
                freq_range=(band.f_min, band.f_max), use_gpu=use_gpu,
                use_tls=True, use_conjugate_pairing=True,
                use_multichannel=False, max_damping=0.2, min_freq=band.f_min,
            )
            n = len(result.frequencies)
            print(f"  {band.name:12s} [{band.f_min:5.0f}-{band.f_max:5.0f} Hz] "
                  f"dec={band.decimation} M={mo:3d} wl={wl} fs_band={fs_band:.0f} "
                  f"samples={len(band_data)} -> {n:3d} modes")
            for i in range(n):
                results.append({
                    'freq': result.frequencies[i],
                    'damping': result.damping_ratios[i],
                    'pole': result.poles[i],
                    'band': band.name,
                    'band_fmin': band.f_min,
                    'band_fmax': band.f_max,
                    'decimation': band.decimation,
                    'model_order': mo,
                    'singular_values': result.singular_values,
                })
        except Exception as e:
            print(f"  {band.name:12s} [{band.f_min:5.0f}-{band.f_max:5.0f} Hz] "
                  f"dec={band.decimation} M={mo:3d} -> FAILED: {e}")

    return results


def find_which_standard_band(freq):
    """Find which STANDARD_BAND(s) would cover this frequency."""
    covering = []
    for b in STANDARD_BANDS:
        if b.f_min <= freq <= b.f_max:
            covering.append(b.name)
    return covering


def find_which_extended_band(freq):
    """Find which EXTENDED_BAND(s) would cover this frequency."""
    covering = []
    for b in EXTENDED_BANDS:
        if b.f_min <= freq <= b.f_max:
            covering.append(b.name)
    return covering


def analyze_svd_comparison(signals, fs, freq_range=(400, 1500)):
    """Compare singular value distributions between configs for the same freq range."""
    print(f"\n{'='*80}")
    print(f"SVD COMPARISON for overlapping frequency range {freq_range[0]}-{freq_range[1]} Hz")
    print(f"{'='*80}")

    # Find STANDARD band covering this range
    for band in STANDARD_BANDS:
        if band.f_min <= freq_range[0] and band.f_max >= freq_range[1]:
            band_data_std, fs_std, _ = process_band(signals, fs, band, apply_preemphasis=True)
            wl = min(2000, len(band_data_std) // 2)
            H_std = build_hankel_matrix(band_data_std[:, 0], wl)
            _, sv_std = esprit_poles(H_std, 30, 1.0/fs_std, use_gpu=True)
            print(f"\n  STANDARD '{band.name}' [{band.f_min}-{band.f_max}Hz] dec={band.decimation}")
            print(f"    Hankel shape: {H_std.shape}, fs_band={fs_std:.0f}")
            print(f"    Top 20 singular values (normalized):")
            sv_n = sv_std / sv_std[0]
            for j in range(min(20, len(sv_n))):
                bar = '#' * int(sv_n[j] * 40)
                print(f"      SV[{j:2d}] = {sv_n[j]:.4f}  {bar}")
            break

    # Find EXTENDED bands covering this range
    for band in EXTENDED_BANDS:
        if band.f_max > freq_range[0] and band.f_min < freq_range[1]:
            band_data_ext, fs_ext, _ = process_band(signals, fs, band, apply_preemphasis=True)
            wl = min(2000, len(band_data_ext) // 2)
            H_ext = build_hankel_matrix(band_data_ext[:, 0], wl)
            mo = band.model_order or 30
            _, sv_ext = esprit_poles(H_ext, mo, 1.0/fs_ext, use_gpu=True)
            print(f"\n  EXTENDED '{band.name}' [{band.f_min}-{band.f_max}Hz] dec={band.decimation}")
            print(f"    Hankel shape: {H_ext.shape}, fs_band={fs_ext:.0f}")
            print(f"    Top 20 singular values (normalized):")
            sv_n = sv_ext / sv_ext[0]
            for j in range(min(20, len(sv_n))):
                bar = '#' * int(sv_n[j] * 40)
                print(f"      SV[{j:2d}] = {sv_n[j]:.4f}  {bar}")


def run_nodecimation_test(signals, fs):
    """Test STANDARD_BANDS but without decimation to see if that's the cause."""
    print(f"\n{'='*80}")
    print("NO-DECIMATION TEST: STANDARD_BANDS with decimation forced to 1")
    print(f"{'='*80}")

    nodec_bands = []
    for b in STANDARD_BANDS:
        nodec_bands.append(FrequencyBand(
            f_min=b.f_min, f_max=b.f_max, filter_order=b.filter_order,
            decimation=1, exp_factor=b.exp_factor, name=b.name + "_nodec",
            model_order=b.model_order,
        ))

    results = run_esprit_multiband(signals, fs, nodec_bands,
                                   global_model_order=30, use_gpu=True)
    return results


def main():
    fs = 48000.0
    print("Loading Belarus Scenario 40 data...")
    signals = load_data()
    print(f"  Shape: {signals.shape}, Duration: {signals.shape[0]/fs:.3f}s")

    # =========================================================================
    # 1. Run both configs
    # =========================================================================
    print(f"\n{'='*80}")
    print("STANDARD_BANDS (model_order=30)")
    print(f"{'='*80}")
    std_results = run_esprit_multiband(signals, fs, STANDARD_BANDS,
                                       global_model_order=30, use_gpu=True)

    print(f"\n{'='*80}")
    print("EXTENDED_BANDS (per-band model orders)")
    print(f"{'='*80}")
    ext_results = run_esprit_multiband(signals, fs, EXTENDED_BANDS,
                                       global_model_order=30, use_gpu=True)

    std_freqs = np.array([r['freq'] for r in std_results])
    std_damp = np.array([r['damping'] for r in std_results])
    ext_freqs = np.array([r['freq'] for r in ext_results])
    ext_damp = np.array([r['damping'] for r in ext_results])

    print(f"\nTotal: STANDARD={len(std_freqs)} modes, EXTENDED={len(ext_freqs)} modes")

    # =========================================================================
    # 2. Mode-by-mode comparison
    # =========================================================================
    print(f"\n{'='*80}")
    print("MODE-BY-MODE COMPARISON: Each STANDARD mode vs closest EXTENDED mode")
    print(f"{'='*80}")

    matched_2hz = []
    close_2_10 = []
    no_match = []

    for i, r in enumerate(std_results):
        f = r['freq']
        d = r['damping']
        band = r['band']
        dec = r['decimation']

        if len(ext_freqs) > 0:
            diffs = np.abs(ext_freqs - f)
            j_closest = np.argmin(diffs)
            closest_f = ext_freqs[j_closest]
            closest_d = ext_damp[j_closest]
            diff = diffs[j_closest]
        else:
            closest_f = np.nan
            closest_d = np.nan
            diff = np.inf

        entry = {
            'std_freq': f, 'std_damp': d, 'std_band': band, 'std_dec': dec,
            'ext_freq': closest_f, 'ext_damp': closest_d, 'freq_diff': diff,
            'ext_bands': find_which_extended_band(f),
        }

        if diff < 2.0:
            matched_2hz.append(entry)
        elif diff < 10.0:
            close_2_10.append(entry)
        else:
            no_match.append(entry)

    # Print matched
    print(f"\n--- MATCHED (< 2 Hz): {len(matched_2hz)} modes ---")
    print(f"  {'STD freq':>10s} {'STD damp':>10s} {'EXT freq':>10s} {'diff':>8s} {'STD band':>12s} {'dec':>4s}")
    for e in sorted(matched_2hz, key=lambda x: x['std_freq']):
        print(f"  {e['std_freq']:10.2f} {e['std_damp']:10.5f} {e['ext_freq']:10.2f} "
              f"{e['freq_diff']:8.2f} {e['std_band']:>12s} {e['std_dec']:>4d}")

    # Print close
    print(f"\n--- CLOSE BUT NOT MATCHED (2-10 Hz): {len(close_2_10)} modes ---")
    print(f"  {'STD freq':>10s} {'STD damp':>10s} {'EXT freq':>10s} {'diff':>8s} {'STD band':>12s} {'dec':>4s} {'EXT bands covering':>25s}")
    for e in sorted(close_2_10, key=lambda x: x['std_freq']):
        ext_b = ', '.join(e['ext_bands']) if e['ext_bands'] else 'NONE'
        print(f"  {e['std_freq']:10.2f} {e['std_damp']:10.5f} {e['ext_freq']:10.2f} "
              f"{e['freq_diff']:8.2f} {e['std_band']:>12s} {e['std_dec']:>4d}  {ext_b}")

    # Print no match
    print(f"\n--- NO MATCH (> 10 Hz): {len(no_match)} modes ---")
    print(f"  {'STD freq':>10s} {'STD damp':>10s} {'EXT closest':>12s} {'diff':>8s} "
          f"{'STD band':>12s} {'dec':>4s} {'EXT bands covering':>25s}")
    for e in sorted(no_match, key=lambda x: x['std_freq']):
        ext_b = ', '.join(e['ext_bands']) if e['ext_bands'] else 'GAP!'
        print(f"  {e['std_freq']:10.2f} {e['std_damp']:10.5f} {e['ext_freq']:12.2f} "
              f"{e['freq_diff']:8.2f} {e['std_band']:>12s} {e['std_dec']:>4d}  {ext_b}")

    # =========================================================================
    # 3. Analysis: damping of unmatched modes
    # =========================================================================
    print(f"\n{'='*80}")
    print("DAMPING ANALYSIS OF UNMATCHED MODES")
    print(f"{'='*80}")

    if no_match:
        nm_damps = [e['std_damp'] for e in no_match]
        m_damps = [e['std_damp'] for e in matched_2hz]
        print(f"  Unmatched modes (>{10}Hz diff): mean damping = {np.mean(nm_damps):.5f}, "
              f"median = {np.median(nm_damps):.5f}")
        if m_damps:
            print(f"  Matched modes (<2Hz diff):     mean damping = {np.mean(m_damps):.5f}, "
                  f"median = {np.median(m_damps):.5f}")
        print(f"\n  Damping distribution of unmatched modes:")
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            count = sum(1 for d in nm_damps if d < thresh)
            print(f"    damping < {thresh:.3f}: {count}/{len(nm_damps)}")

    # =========================================================================
    # 4. Decimation analysis
    # =========================================================================
    print(f"\n{'='*80}")
    print("DECIMATION ANALYSIS: Are decimated bands the ones losing modes?")
    print(f"{'='*80}")

    dec_bands = [b for b in STANDARD_BANDS if b.decimation > 1]
    nodec_bands_std = [b for b in STANDARD_BANDS if b.decimation == 1]

    dec_names = {b.name for b in dec_bands}
    nodec_names = {b.name for b in nodec_bands_std}

    nm_from_dec = [e for e in no_match if e['std_band'] in dec_names]
    nm_from_nodec = [e for e in no_match if e['std_band'] in nodec_names]
    close_from_dec = [e for e in close_2_10 if e['std_band'] in dec_names]
    close_from_nodec = [e for e in close_2_10 if e['std_band'] in nodec_names]
    matched_from_dec = [e for e in matched_2hz if e['std_band'] in dec_names]
    matched_from_nodec = [e for e in matched_2hz if e['std_band'] in nodec_names]

    print(f"\n  Decimated bands ({', '.join(dec_names)}):")
    total_dec = len([r for r in std_results if r['band'] in dec_names])
    print(f"    Total modes: {total_dec}")
    print(f"    Matched (<2Hz): {len(matched_from_dec)}")
    print(f"    Close (2-10Hz): {len(close_from_dec)}")
    print(f"    No match (>10Hz): {len(nm_from_dec)}")

    print(f"\n  Non-decimated bands ({', '.join(nodec_names)}):")
    total_nodec = len([r for r in std_results if r['band'] in nodec_names])
    print(f"    Total modes: {total_nodec}")
    print(f"    Matched (<2Hz): {len(matched_from_nodec)}")
    print(f"    Close (2-10Hz): {len(close_from_nodec)}")
    print(f"    No match (>10Hz): {len(nm_from_nodec)}")

    # =========================================================================
    # 5. Tolerance sensitivity
    # =========================================================================
    print(f"\n{'='*80}")
    print("TOLERANCE SENSITIVITY: How overlap changes with tolerance")
    print(f"{'='*80}")

    for tol in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        if len(ext_freqs) > 0:
            overlap = sum(1 for f in std_freqs if np.any(np.abs(ext_freqs - f) < tol))
        else:
            overlap = 0
        pct = 100 * overlap / len(std_freqs) if len(std_freqs) > 0 else 0
        print(f"  Tolerance {tol:5.1f} Hz: {overlap:3d}/{len(std_freqs)} "
              f"STANDARD modes matched ({pct:.1f}%)")

    # =========================================================================
    # 6. Band coverage gaps
    # =========================================================================
    print(f"\n{'='*80}")
    print("BAND COVERAGE: Where EXTENDED_BANDS has gaps or narrow overlaps")
    print(f"{'='*80}")

    print(f"\n  STANDARD_BANDS coverage:")
    for b in STANDARD_BANDS:
        print(f"    {b.name:12s}: {b.f_min:6.0f} - {b.f_max:6.0f} Hz  "
              f"(dec={b.decimation}, M={b.model_order or 30})")

    print(f"\n  EXTENDED_BANDS coverage:")
    for b in EXTENDED_BANDS:
        print(f"    {b.name:12s}: {b.f_min:6.0f} - {b.f_max:6.0f} Hz  "
              f"(dec={b.decimation}, M={b.model_order or 30})")

    # Check for modes falling near band boundaries
    print(f"\n  Unmatched modes near EXTENDED band boundaries (within 20 Hz of edge):")
    for e in sorted(no_match + close_2_10, key=lambda x: x['std_freq']):
        f = e['std_freq']
        near_edges = []
        for b in EXTENDED_BANDS:
            if abs(f - b.f_min) < 20:
                near_edges.append(f"near {b.name} lower edge ({b.f_min} Hz)")
            if abs(f - b.f_max) < 20:
                near_edges.append(f"near {b.name} upper edge ({b.f_max} Hz)")
        if near_edges:
            print(f"    {f:8.2f} Hz: {'; '.join(near_edges)}")

    # =========================================================================
    # 7. No-decimation test
    # =========================================================================
    nodec_results = run_nodecimation_test(signals, fs)
    nodec_freqs = np.array([r['freq'] for r in nodec_results])

    print(f"\n  No-decimation STANDARD: {len(nodec_freqs)} modes (vs {len(std_freqs)} with decimation)")

    # How many of the original modes are recovered?
    if len(nodec_freqs) > 0:
        for tol in [2.0, 5.0, 10.0]:
            overlap_nd_vs_std = sum(1 for f in std_freqs if np.any(np.abs(nodec_freqs - f) < tol))
            overlap_nd_vs_ext = sum(1 for f in nodec_freqs if np.any(np.abs(ext_freqs - f) < tol))
            print(f"    Tol={tol:.0f}Hz: {overlap_nd_vs_std}/{len(std_freqs)} std modes in no-dec, "
                  f"{overlap_nd_vs_ext}/{len(nodec_freqs)} no-dec modes in extended")

    # =========================================================================
    # 8. SVD comparison
    # =========================================================================
    analyze_svd_comparison(signals, fs, freq_range=(400, 1500))

    # =========================================================================
    # 9. Key insight: model order vs modes per band
    # =========================================================================
    print(f"\n{'='*80}")
    print("MODEL ORDER vs EXTRACTED MODES: Is model_order=30 extracting phantom modes?")
    print(f"{'='*80}")

    print(f"\n  STANDARD config uses global model_order=30 for all bands.")
    print(f"  EXTENDED uses per-band model orders (10-50).")
    print(f"\n  In STANDARD, Low band (30-200Hz, dec=4, fs=12000):")
    print(f"    model_order=30 means 30 poles = up to 15 conjugate pairs = 15 modes")
    print(f"    But the 30-200Hz range has maybe 5-8 physical modes")
    print(f"    -> OVERFIT: many spurious modes accepted!")
    print(f"\n  In EXTENDED, Ultra-Low (30-100Hz, dec=1, fs=48000):")
    print(f"    model_order=10 means 10 poles = up to 5 conjugate pairs = 5 modes")
    print(f"    -> more conservative, fewer spurious modes")

    # Count modes per Hz-width for each config
    print(f"\n  Mode density (modes per 100 Hz bandwidth):")
    print(f"  {'Band':>20s} {'BW (Hz)':>8s} {'Modes':>6s} {'Density':>10s}")
    for band in STANDARD_BANDS:
        bw = band.f_max - band.f_min
        modes_in_band = sum(1 for r in std_results if r['band'] == band.name)
        density = modes_in_band / bw * 100
        print(f"  STD {band.name:>14s} {bw:8.0f} {modes_in_band:6d} {density:10.2f}")
    for band in EXTENDED_BANDS:
        bw = band.f_max - band.f_min
        modes_in_band = sum(1 for r in ext_results if r['band'] == band.name)
        density = modes_in_band / bw * 100
        print(f"  EXT {band.name:>14s} {bw:8.0f} {modes_in_band:6d} {density:10.2f}")

    # =========================================================================
    # 10. Reverse analysis: EXTENDED modes not in STANDARD
    # =========================================================================
    print(f"\n{'='*80}")
    print("REVERSE: EXTENDED modes NOT found in STANDARD")
    print(f"{'='*80}")

    ext_only = []
    for r in ext_results:
        f = r['freq']
        if len(std_freqs) > 0:
            diff = np.min(np.abs(std_freqs - f))
        else:
            diff = np.inf
        if diff > 10.0:
            ext_only.append({'freq': f, 'damping': r['damping'], 'band': r['band'],
                             'closest_std': std_freqs[np.argmin(np.abs(std_freqs - f))] if len(std_freqs) > 0 else np.nan,
                             'diff': diff})

    print(f"  {len(ext_only)} EXTENDED modes with no STANDARD match (>10Hz)")
    if ext_only:
        print(f"  {'EXT freq':>10s} {'EXT damp':>10s} {'Band':>12s} {'Closest STD':>12s} {'diff':>8s}")
        for e in sorted(ext_only, key=lambda x: x['freq'])[:30]:
            print(f"  {e['freq']:10.2f} {e['damping']:10.5f} {e['band']:>12s} "
                  f"{e['closest_std']:12.2f} {e['diff']:8.2f}")
        if len(ext_only) > 30:
            print(f"  ... and {len(ext_only) - 30} more")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY OF FINDINGS")
    print(f"{'='*80}")
    print(f"""
  STANDARD_BANDS: {len(std_freqs)} modes  |  EXTENDED_BANDS: {len(ext_freqs)} modes

  Overlap at various tolerances:""")
    for tol in [2.0, 5.0, 10.0]:
        if len(ext_freqs) > 0:
            ov = sum(1 for f in std_freqs if np.any(np.abs(ext_freqs - f) < tol))
        else:
            ov = 0
        print(f"    {tol:4.0f} Hz: {ov}/{len(std_freqs)} ({100*ov/max(len(std_freqs),1):.0f}%)")

    print(f"""
  Unmatched STANDARD modes: {len(no_match)} (>{10}Hz from any EXTENDED mode)
    From decimated bands: {len(nm_from_dec)}
    From non-decimated bands: {len(nm_from_nodec)}
    Mean damping (unmatched): {np.mean([e['std_damp'] for e in no_match]) if no_match else 0:.5f}
    Mean damping (matched):   {np.mean([e['std_damp'] for e in matched_2hz]) if matched_2hz else 0:.5f}
""")


if __name__ == "__main__":
    main()
