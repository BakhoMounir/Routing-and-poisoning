# 04 - Provenance and license: what only you can resolve (not legal advice)

Known (PROVENANCE.md): `Network_nsfnetbw.ned` (SHA-256 fe497b0b...) comes from a third-party GitHub fork; the fork's BSD-3 text covers its code, not necessarily the data; the official BNN-UPC archive was unreachable; no dataset license was found; the 500 samples are packet-level *simulation* outputs.
Nothing in the supplied files can settle identity or license. Suggested actions:
1. Obtain the official NSFNET-bw archive (BNN-UPC / KDN dataset page) from your network, compare `Network_nsfnetbw.ned` byte-for-byte or by parsed topology; record the result and URL.
2. Ask the dataset authors (or check their site/paper) for the data license; record the answer.
3. Until resolved: do not publish `data/raw`; describe the topology as "NSFNET-bw as distributed with RouteNet-style datasets (identity unverified)"; describe traffic as synthetic and total-volume-calibrated (see audit item 7).
4. If a derived edge list is to be published, decide that with the license in hand.
The new scripts refuse to run unless the NED hash equals the recorded one, so a swapped file cannot silently change results. If you legitimately replace the NED with the verified official one, update `topology.sha256` deliberately (and treat all earlier results as a different topology).
