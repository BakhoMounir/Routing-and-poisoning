# 04 - Provenance and license: what only you can resolve (not legal advice)

Known (PROVENANCE.md): `Network_nsfnetbw.ned` (SHA-256 fe497b0b...) comes from a third-party GitHub fork; the fork's BSD-3 text covers its code, not necessarily the data; the official BNN-UPC archive was unreachable; no dataset license was found; the 500 samples are packet-level *simulation* outputs.
Nothing in the supplied files can settle identity or license. Suggested actions:
1. Obtain the official NSFNET-bw archive (BNN-UPC / KDN dataset page) from your network, compare `Network_nsfnetbw.ned` byte-for-byte or by parsed topology; record the result and URL.
2. Ask the dataset authors (or check their site/paper) for the data license; record the answer.
3. Until resolved: do not publish `data/raw`; describe the topology as "NSFNET-bw as distributed with RouteNet-style datasets (identity unverified)"; describe traffic as synthetic and total-volume-calibrated (see audit item 7).
4. If a derived edge list is to be published, decide that with the license in hand.
The new scripts refuse to run unless the NED hash equals the recorded one, so a swapped file cannot silently change results. If you legitimately replace the NED with the verified official one, update `topology.sha256` deliberately (and treat all earlier results as a different topology).

## Confirmatory evaluation artifact audit

The later clean-stage confirmatory run is recorded under
`results/v2/confirmatory_v1/` in the external handover directory. Its final
artifact audit reported 35 checks passed and 0 failed. The evaluation manifest
records 72,000 episode rows, a completed status, and `git: null`.

The audit verifies the recorded freeze/configuration relationship, episode
CSV hash, episode-level invariants, scenario coverage, and analysis-output
hashes. This is an artifact-integrity and consistency result, not a resolution
of dataset identity, licensing, or redistribution rights.

The episode CSV SHA-256 is
`e5257ccfa3e8b56661f35833fc894ee9b0bda06c2adf79f25552d442d7f83b73`.
The frozen configuration SHA-256 is
`2c2a530d649f947b791d33c7c0de58ad9da0e138e9532ca159cf1b24e7eb34ec`.

Do not publish the raw dataset while its rights remain unresolved. Keep the
evaluation outputs identified as external handover artifacts unless and until
they are deliberately added to a versioned, documented release.
