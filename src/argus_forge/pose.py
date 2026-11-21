"""The pose-dominance warning, read from curator's manifest (2.7+).

argus-curator asks argus-puppet for each image's body pose and writes two plain
values per manifest row: ``pose_cluster`` (a scan-level cluster id) and
``pose_labels`` (``{stance, arms, facing}``). When most of a training set sits in
one pose cluster, a LoRA tends to collapse onto that pose, so forge warns.

Forge only **counts** the ids and labels curator wrote. It never measures a pose,
compares two, or calls puppet: that is puppet's job, and a second copy of its
metrics here would drift (see the must-never table in CLAUDE.md). Repeats,
heuristics and emitters are untouched; this is a warning and nothing else.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from argus_forge.models import ManifestRow

# Fewer clustered rows than this and a "dominant" cluster is noise: no warning.
POSE_MIN_CLUSTERED = 10

# The share of clustered rows in the most common cluster at which forge warns.
# argus-puppet's and argus-proof's pose-collapse cut-off, kept the same so the
# suite gives one answer. Uncalibrated on real training sets: a starting point.
POSE_DOMINANCE_SHARE = 0.75

# Rows with no cluster (unusable pose, nobody found, not checked) are left out of
# the share. When they are at least this share of the set, the warning says how
# many, so "N of M" is not read as the whole set.
POSE_UNMEASURED_NOTE_SHARE = 0.25

# The label value puppet writes when the joints a rule needs were not confident.
UNKNOWN_LABEL = "unknown"


def _label_phrase(rows: list[ManifestRow]) -> str:
    """The most common non-``unknown`` value of each label among *rows*, as a phrase, or ''."""
    parts: list[str] = []
    for field, template in (("stance", "{}"), ("arms", "arms {}"), ("facing", "facing {}")):
        values = Counter(
            value
            for row in rows
            if row.pose_labels is not None
            and (value := getattr(row.pose_labels, field))
            and value.strip()
            and value != UNKNOWN_LABEL
        )
        if values:
            parts.append(template.format(values.most_common(1)[0][0]))
    return ", ".join(parts)


def pose_dominance_warning(rows: Iterable[ManifestRow]) -> str | None:
    """One warning when one body-pose cluster dominates *rows*, else None.

    *rows* are the manifest rows of the images being trained. Rows without a
    ``pose_cluster`` (older manifests, unusable or unchecked poses) are not
    counted; with fewer than :data:`POSE_MIN_CLUSTERED` clustered rows there is
    no warning. Curator's clusters were formed over its whole scan, so the ids
    are compared as given: two rows in the same cluster share a pose in that
    scan's terms whatever else was exported.
    """
    rows = list(rows)
    clustered = [row for row in rows if row.pose_cluster is not None]
    if len(clustered) < POSE_MIN_CLUSTERED:
        return None
    cluster, size = Counter(row.pose_cluster for row in clustered).most_common(1)[0]
    if size / len(clustered) < POSE_DOMINANCE_SHARE:
        return None

    labels = _label_phrase([row for row in clustered if row.pose_cluster == cluster])
    pose = f"one body pose ({labels})" if labels else "one body pose"
    note = (
        f"{size} of {len(clustered)} posed training images share {pose}; outputs are likely to collapse onto it "
        "— rebalance the selection in curator"
    )
    unmeasured = len(rows) - len(clustered)
    if unmeasured and unmeasured / len(rows) >= POSE_UNMEASURED_NOTE_SHARE:
        note += f" ({unmeasured} of {len(rows)} images had no measured pose and are not counted)"
    return note
